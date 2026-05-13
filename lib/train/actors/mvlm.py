from __future__ import annotations

"""
MVLM_Actor (patched)
=======================
* **Fix 2025‑04‑28**: `compute_losses` now passes `gt_boxes_rep` (shape B×N×4) to
  `match_boxes` and `loss_bbox`. Previous bug passed the flattened `gt_boxes_vec`,
  causing shape mismatch (4 vs 32) at L1.
* Box handling, matching, and losses remain as in the 1st patch (coord‑sort +
  clamp).
"""

from . import BaseActor
from lib.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_xywh_to_xyxy,
)
from lib.utils.heapmap_utils import generate_heatmap
import torch
import torch.nn as nn


class MVLM_Actor(BaseActor):
    def __init__(self, net, objective: Dict[str, nn.Module], loss_weight: Dict[str, float], settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = settings.batchsize
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        self.template_free = cfg.MODEL.ENCODER.TEMPLATE_FREE
        self.num_template = cfg.DATA.TEMPLATE.NUMBER

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        b = data["search_images"].shape[1]
        search_list = data["search_images"].reshape(-1, *data["search_images"].shape[2:]).split(b, dim=0)
        template_list = data["template_images"].reshape(-1, *data["template_images"].shape[2:]).split(b, dim=0)
        raw_text = data["nlp_real"] if self.multi_modal_language else None
        
        outputs = self.net(
            template_list=template_list,
            search_list=search_list,
            text_emb=None,
            raw_text=raw_text,
        )

        return outputs


    def tracking_loss(self, pred_dict, gt_dict):

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.ENCODER.STRIDE)  # list of torch.Size([b, H, W])
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # torch.Size([b, 1, H, W])

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']  # torch.Size([b, 1, 4])
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

        gt_boxes_vec = (
            box_xywh_to_xyxy(gt_bbox)[:, None, :]
            .repeat((1, num_queries, 1))
            .view(-1, 4)
            .clamp(min=0.0, max=1.0)
        )  # (B,4) --> (B,1,4) --> (B,N,4)
        
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss
                )

        return loss, iou
    
    
    def cm_loss(self, pred_dict, gt_dict):
        # compute cm loss
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        tae = pred_dict['tae']
        
        lang_token_map = tae[:, -77:, :]
        if self.template_free:
            vis_token_map = tae[:, :-77, :]
        else:
            vis_token_map = tae[:, :-77 - 49 * self.num_template, :]
        
        if 'cm' in self.objective:
            cm_loss_raw = self.objective['cm'](lang_token_map, vis_token_map, gt_bbox, 1.0, 1.0)
        else:
            cm_loss_raw = torch.tensor(0.0, device=gt_bbox.device)
        return cm_loss_raw, self.loss_weight['cm'] * cm_loss_raw


    def compute_losses(self, pred_dict: Dict[str, torch.Tensor], gt_dict: Dict[str, torch.Tensor], *,
                       return_status=True):

        tracking_loss, ious = self.tracking_loss(pred_dict, gt_dict)
        
        if self.cfg.TRAIN.CM_LOSS:
            cm_loss_raw, cm_loss_weighted = self.cm_loss(pred_dict, gt_dict)
        else:
            cm_loss_raw = torch.tensor(0.0, device=pred_dict['score_map'].device)
            cm_loss_weighted = torch.tensor(0.0, device=pred_dict['score_map'].device)

        loss = (
                tracking_loss +
                cm_loss_weighted
        )

        if not return_status:
            return loss
        status = {
            "Loss/Total": loss.item(),
            "Loss/Tracking": tracking_loss.item(),
            "IoU": ious.mean().item()}
        if self.cfg.TRAIN.CM_LOSS:
            status["Loss/CM"] = cm_loss_weighted.item()  # After applying weight
        return loss, status   