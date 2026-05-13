from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F
from lib.test.tracker.utils import Preprocessor, sample_target, transform_image_to_crop, resize_sample_target
import cv2
from lib.test.utils.hann import hann2d
from lib.models.mvlm import build_mvlm
from lib.utils.box_ops import clip_box


class MVLM(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MVLM, self).__init__(params)
        network = build_mvlm(params.cfg)
        ckpt = torch.load(self.params.checkpoint, map_location='cpu')
        state_dict = ckpt['net'] if 'net' in ckpt else ckpt
        network.load_state_dict(state_dict, strict=False)
        self.cfg = params.cfg
        if hasattr(params, 'device') and params.device:
            self.device = torch.device(params.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = network.to(self.device)
        self.network.eval()
        self.preprocessor = Preprocessor(self.device)
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).to(self.device)

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0
        
        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT

        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT

        # mapping similar datasets
        if 'LASOT' in DATASET_NAME:
            DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME:
            DATASET_NAME = 'TNL2K'

        # multi modal vision
        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT

        #multi modal language
        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT

        #using nlp information
        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT

        self.prev_conf = 1.0
        
        self.template_free = self.cfg.MODEL.ENCODER.TEMPLATE_FREE
        
        self.omega = self.cfg.TEST.MVLM.OMEGA
        self.lambda_mem = self.cfg.TEST.MVLM.LAMBDA

        self.alpha_corr = self.cfg.TEST.MVLM.ALPHA_CORR
        self.alpha_cls = 1 - self.alpha_corr
        self.iou_thresh = self.cfg.TEST.MVLM.PSI_OUT
        self.tau = self.cfg.TEST.MVLM.TAU

    def initialize(self, image, info: dict):
        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and (template.size(1) == 3):
            template = torch.cat((template, template), axis=1)
        self.template_list = [template] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)]
        self.frame_id = 0
        
        # language information
        if self.multi_modal_language:
            if self.use_nlp:
                self.init_nlp = info['init_nlp']
            else:
                self.init_nlp = None
            with torch.no_grad():
                self.text_emb = self.network.text_encoder(self.init_nlp)
        else:
            self.text_emb = None
            self.init_nlp = None
        
        if self.template_free:
            self.flag = True
        else:
            self.flag = False

        self.kmem = None
        self.kmvlm = None

    def track(self, image, info: dict = None):

        H, W, _ = image.shape
        self.frame_id += 1
            
        if self.flag:
            x_patch_arr, resize_factor = resize_sample_target(image, output_sz=self.params.search_size)  # (x1, y1, w, h)
            # crop region covers the entire image
        else:
            x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor, output_sz=self.params.search_size)  # (x1, y1, w, h)
            # compute crop region in original image coordinates
            x, y, w, h = self.state

        search = self.preprocessor.process(x_patch_arr)
        if self.multi_modal_vision and (search.size(1) == 3):
            search = torch.cat((search, search), axis=1)
        search_list = [search]

        # run the encoder
        with torch.no_grad():
            out_dict = self.network(template_list=self.template_list,
                                    search_list=search_list,
                                    text_emb=self.text_emb,
                                    raw_text=[self.init_nlp] if self.init_nlp is not None else None,
                                    )

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True: # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
            
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.head.cal_bboxv2(response, out_dict['size_map'],
                                                                  out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.head.cal_bboxv2(response,
                                                                  out_dict['offset_map'],
                                                                  return_score=True)

        boxes = pred_boxes[0]       # [N, 4] (cx, cy, w, h)
        cls_scores = conf_score[0]  # [N]
        
        # ── VL correlation scores s(I, t, b) for each candidate box ──
        vis_token = torch.cat([out_dict['feat_x']], dim=1)
        feats = batched_slice_mean_from_grid(vis_token, boxes)  # [N, C]

        text_vec = self.text_emb.mean(dim=1).squeeze(0)  # [C]
        feats = F.normalize(feats, p=2, dim=-1)
        text_vec = F.normalize(text_vec, p=2, dim=-1)
        sims = (feats * text_vec.unsqueeze(0)).sum(dim=-1)  # [N]

        # ── Per-box distractor selection (Eq. 4-5 in paper) ──
        # For each box b, find b_tilde_corr(b) and b_tilde_cls(b) from B^out(b)
        distract_corr_scores, distract_cls_scores = select_per_box_distractor(
            boxes, sims, cls_scores, iou_thr=self.iou_thresh
        )  # both [N]

        # ── Per-box standardized margins with tanh (Eq. 4-5) ──
        delta_corr = torch.tanh(sims - distract_corr_scores)          # [N], Eq. (4)
        delta_cls  = torch.tanh(cls_scores - distract_cls_scores)      # [N], Eq. (5)

        # ── Per-box VLM confidence (Eq. 6) ──
        kvlm_per_box = self.alpha_corr * delta_corr + self.alpha_cls * delta_cls  # [N]

        # ── Per-token EWMA temporal memory (Eq. 7-8) ──
        # Grid Omega is fixed across frames → per-token correspondence maintained
        if self.kmem is None:
            self.kmem = kvlm_per_box.clone()   # kappa^mem_0(b) := kappa^vlm_1(b)
        else:
            self.kmem = (1 - self.lambda_mem) * kvlm_per_box + self.lambda_mem * self.kmem  # Eq. (8)

        # ── Per-box MVLM confidence (Eq. 9) ──
        kmvlm_per_box = (1 - self.omega) * kvlm_per_box + self.omega * self.kmem  # [N]

        # ── ROI selection S_t(tau) (Eq. 10) ──
        selected_boxes = kmvlm_per_box >= self.tau

        if torch.any(selected_boxes):
            # |S_t| > 0: select b_hat = argmax_{b in S_t} l_t(b), local search next frame
            cand_idx = torch.nonzero(selected_boxes, as_tuple=False).squeeze(1)
            cand_confs = cls_scores[cand_idx]
            best_rel = torch.argmax(cand_confs)
            top1_idx = int(cand_idx[best_rel].item())
            next_flag = False
        else:
            # |S_t| = 0: select b_hat = argmax_{b in B_t} l_t(b), global re-localization next frame
            top1_idx = int(torch.argmax(cls_scores).item())
            next_flag = True

        top1_box = pred_boxes[0][top1_idx]

        if self.flag:
            box = (top1_box * self.params.search_size).tolist()  # (cx, cy, w, h)
            # resize_sample_target resizes (W, H) → (search_size, search_size)
            # per-axis scale factors needed for non-square images
            if isinstance(resize_factor, (tuple, list)):
                w_scale, h_scale = resize_factor
            else:
                w_scale = self.params.search_size / W
                h_scale = self.params.search_size / H
            cx, cy, w, h = box
            cx_origin = cx / w_scale
            cy_origin = cy / h_scale
            w_origin = w / w_scale
            h_origin = h / h_scale
            pred_box = [cx_origin - 0.5 * w_origin, cy_origin - 0.5 * h_origin, w_origin, h_origin]
            self.state = clip_box(pred_box, H, W, margin=10)
        else:
            box = top1_box.view(-1, 4)
            pred_box = (box.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # LOCAL OR GLOBAL VIEW ON NEXT FRAME?
        if next_flag:
            self.flag = True
        else:
            self.flag = False
        
        if not self.template_free and self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (cls_scores[top1_idx] > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                if self.multi_modal_vision and (template.size(1) == 3):
                    template = torch.cat((template, template), axis=1)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        # for debug
        if self.debug:
            image_show = image
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            image_BGR = cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
        else:
            image_BGR = None

        if self.debug:
            # visualize prediction on cropped search image
            cropped_vis = x_patch_arr.copy()
            sz = self.params.search_size
            cx_c, cy_c, w_c, h_c = (top1_box * sz).tolist()
            x1_c = int(cx_c - 0.5 * w_c)
            y1_c = int(cy_c - 0.5 * h_c)
            x2_c = int(cx_c + 0.5 * w_c)
            y2_c = int(cy_c + 0.5 * h_c)
            cv2.rectangle(cropped_vis, (x1_c, y1_c), (x2_c, y2_c), color=(0, 0, 255), thickness=2)
        else:
            cropped_vis = None

        results_dict = {
            "target_bbox": self.state,
            "conf_score": conf_score,
            "image_BGR": image_BGR,
            "cropped_vis": cropped_vis,
        }

        return results_dict

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


# ─────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────

def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)


def pairwise_iou(boxes):
    """Compute NxN pairwise IoU matrix for boxes in (cx,cy,w,h) format.
    
    Args:
        boxes: [N, 4] tensor in (cx, cy, w, h) format
    Returns:
        iou_matrix: [N, N] tensor
    """
    xyxy = cxcywh_to_xyxy(boxes)  # [N, 4]
    x1 = torch.maximum(xyxy[:, None, 0], xyxy[None, :, 0])  # [N, N]
    y1 = torch.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    x2 = torch.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    y2 = torch.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # [N, N]
    area = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)  # [N]
    union = area[:, None] + area[None, :] - inter + 1e-12
    return inter / union


def select_per_box_distractor(boxes, corr_scores, cls_scores, iou_thr=0.3):
    """For each box b, find its strongest distractor from B^out(b).
    
    Paper Sec 3.3:
        B^out_t(b) = { b' in B_t : IoU(b', b) <= psi_out }
        b_tilde_corr(b) = argmax_{b' in B^out(b)} s(I, t, b')
        b_tilde_cls(b)  = argmax_{b' in B^out(b)} l_t(b')
    
    Args:
        boxes:       [N, 4] candidate boxes (cx, cy, w, h)
        corr_scores: [N]    VL correlation scores s(I, t, b)
        cls_scores:  [N]    classification scores l_t(b)
        iou_thr:     float  psi_out threshold
    Returns:
        distract_corr_scores: [N]  s(I, t, b_tilde_corr(b)) for each b
        distract_cls_scores:  [N]  l_t(b_tilde_cls(b)) for each b
    """
    N = boxes.size(0)
    iou_matrix = pairwise_iou(boxes)  # [N, N]

    # B^out(b) mask: IoU(b', b) <= psi_out, excluding self
    outside_mask = iou_matrix <= iou_thr  # [N, N]
    outside_mask.fill_diagonal_(False)    # exclude b itself

    # For correlation: find max s(I,t,b') within B^out(b) for each b
    corr_expanded = corr_scores.unsqueeze(0).expand(N, -1)  # [N, N] — row i = scores of all b'
    corr_masked = corr_expanded.clone()
    corr_masked[~outside_mask] = -float('inf')
    distract_corr_scores = corr_masked.max(dim=1).values  # [N]

    # For classification: find max l_t(b') within B^out(b) for each b
    cls_expanded = cls_scores.unsqueeze(0).expand(N, -1)  # [N, N]
    cls_masked = cls_expanded.clone()
    cls_masked[~outside_mask] = -float('inf')
    distract_cls_scores = cls_masked.max(dim=1).values  # [N]

    # Fallback: if B^out(b) is empty for some b, use min score as distractor
    no_outside_corr = (corr_masked > -float('inf')).sum(dim=1) == 0
    no_outside_cls  = (cls_masked  > -float('inf')).sum(dim=1) == 0
    if no_outside_corr.any():
        distract_corr_scores[no_outside_corr] = corr_scores.min()
    if no_outside_cls.any():
        distract_cls_scores[no_outside_cls] = cls_scores.min()

    return distract_corr_scores, distract_cls_scores


def batched_slice_mean_from_grid(vis_token, boxes_cxcywh):
    if vis_token.dim() == 3:
        vis_token = vis_token[0]
    L, C = vis_token.shape
    H = W = int(L ** 0.5)
    feat = vis_token.view(H, W, C)

    Scale = (H - 1)
    cx, cy, w, h = boxes_cxcywh.unbind(-1)

    x1 = torch.round((cx - 0.5 * w) * Scale).to(torch.int64)
    y1 = torch.round((cy - 0.5 * h) * Scale).to(torch.int64)
    x2 = torch.round((cx + 0.5 * w) * Scale).to(torch.int64)
    y2 = torch.round((cy + 0.5 * h) * Scale).to(torch.int64)

    x1 = x1.clamp(min=0)
    y1 = y1.clamp(min=0)
    x2 = torch.maximum(x2, x1 + 1)
    y2 = torch.maximum(y2, y1 + 1)

    x1 = x1.clamp(max=W - 1)
    y1 = y1.clamp(max=H - 1)
    x2 = x2.clamp(max=W)
    y2 = y2.clamp(max=H)

    zero_pad = feat.new_zeros((1, W, C))
    S = torch.cat([zero_pad, feat.cumsum(dim=0)], dim=0)  # [H+1, W, C]
    zero_pad2 = S.new_zeros((H + 1, 1, C))
    S = torch.cat([zero_pad2, S.cumsum(dim=1)], dim=1)  # [H+1, W+1, C]

    x1i = x1 + 1
    y1i = y1 + 1
    x2i = x2
    y2i = y2

    rect_sum = S[y2i, x2i] - S[y1i, x2i] - S[y2i, x1i] + S[y1i, x1i]
    area = (y2 - y1) * (x2 - x1)
    area = area.clamp(min=1).unsqueeze(1)

    feats = rect_sum / area
    return feats


def get_tracker_class():
    return MVLM