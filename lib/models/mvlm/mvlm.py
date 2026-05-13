import torch
import math
from torch import nn
from .encoder import build_encoder
from .head import build_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.mvlm.language.clip import build_clip_textencoder


class MVLM(nn.Module):
    def __init__(self, text_encoder, encoder, head,
                 num_frames=1, num_template=1,
                 head_type="CENTER"):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.head_type = head_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.head = head

        self.num_frames = num_frames
        self.num_template = num_template


    def forward(self, template_list=None, search_list=None, text_emb=None, raw_text=None):

        auc_dict = {}

        if self.text_encoder is not None:
            if text_emb is None and raw_text is not None:
                text_emb = self.forward_textencoder(raw_text)
                auc_dict.update({"text_emb": text_emb})
        else:
            text_emb = None

        enc_dict = self.forward_encoder(template_list, search_list, text_emb)
        head_dict = self.forward_head(enc_dict['feat_x'])
        auc_dict.update(enc_dict)
        auc_dict.update(head_dict)

        return auc_dict

    def forward_textencoder(self, raw_text):
        # Forward the encoder
        text_emb = self.text_encoder(raw_text)
        return text_emb

    def forward_encoder(self, template_list, search_list, text_emb):
        enc_dict = self.encoder(template_list, search_list, text_emb)
        return enc_dict

    def forward_head(self, feature, gt_score_map=None):

        bs, HW, C = feature.size()
        if self.head_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.head(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.head(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.head_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.head(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def load_pretrained_tracker(model, pretrained):
    # load state_dict of the pretrained model except the text encoder
    checkpoint = torch.load(pretrained, map_location="cpu")
    state_dict = checkpoint['net']

    keys_to_remove = [k for k in state_dict.keys() if 'text_encoder' in k]
    for key in keys_to_remove:
        del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    return model

def build_mvlm(cfg):
    encoder = build_encoder(cfg)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        if cfg.MODEL.ENCODER.LANGUAGE_MODEL == 'CLIP':
            text_encoder = build_clip_textencoder(cfg, encoder) # CLIP
        else:
            print('Non-expected Language Model!')
            exit(-1)
    else:
        text_encoder = None
    head = build_head(cfg, encoder)
    model = MVLM(
        text_encoder,
        encoder,
        head,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
    
    if cfg.MODEL.PRETRAINED_MODEL is not None:
        model = load_pretrained_tracker(model, cfg.MODEL.PRETRAINED_MODEL)
    
    return model
