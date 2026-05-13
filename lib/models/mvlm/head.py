import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.utils.box_ops import box_xyxy_to_cxcywh

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid (use register_buffer for automatic device transfer)
            self.register_buffer('coord_x', self.indice.repeat((self.feat_sz, 1))
                .view((self.feat_sz * self.feat_sz,)).float())
            self.register_buffer('coord_y', self.indice.repeat((1, self.feat_sz))
                .view((self.feat_sz * self.feat_sz,)).float())

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16,
                 conv_type='normal', freeze_bn=False, xavier_init=True):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        if conv_type == "normal":
            kernel_size = 3
            padding = 1
        elif conv_type == "small":
            kernel_size = 1
            padding = 0
        else:
            raise NotImplementedError('cfg.MODEL.HEAD.CONV_TYPE must be choosen from "normal" and "small".')

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn, kernel_size=kernel_size, padding=padding)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        if xavier_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x) # x: torch.Size([b, c, h, w])

        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True) # score_map_ctr.flatten(1): torch.Size([32, 256]) idx: torch.Size([32, 1]) max_score: torch.Size([32, 1])
        idx_y = torch.div(idx, self.feat_sz, rounding_mode='floor')
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx) # size_map: torch.Size([32, 2, 16, 16])  size_map.flatten(2): torch.Size([32, 2, 256])
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox
    
    def cal_bboxv2(self, score_map_ctr, size_map, offset_map, return_score=True):
        B, _, H, W = score_map_ctr.shape
        device, dtype = score_map_ctr.device, torch.float32
        score_flat = score_map_ctr.view(B, -1)  # [B, HW]
        all_bboxes, all_scores = [], []
        ys, xs = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device),
                                indexing="ij")  # ys: [H,W], xs:[H,W]
        xs = xs.reshape(-1)  # [HW]
        ys = ys.reshape(-1)  # [HW]
        for b in range(B):
            # [2, HW]
            size_b = size_map[b].view(2, -1)
            offset_b = offset_map[b].view(2, -1)
            cx = (xs.to(dtype) + offset_b[0]) / float(W)  # [HW]
            cy = (ys.to(dtype) + offset_b[1]) / float(H)  # [HW]
            w, h = size_b[0], size_b[1]  # [HW]
            bboxes_b = torch.stack([cx, cy, w, h], dim=1).to(device=device, dtype=dtype)  # [HW,4]
            scores_sel = score_flat[b]  # [HW]
            all_bboxes.append(bboxes_b)
            all_scores.append(scores_sel) 
        if return_score:
            return all_bboxes, all_scores
        else:
            return all_bboxes

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

class MLPPredictor(nn.Module):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(MLPPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        self.num_layers = 3
        h = [channel] * (self.num_layers - 1)
        self.layers_cls = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([inplanes] + h, h + [1]))
        self.layers_reg = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([inplanes] + h, h + [4]))


    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), offset_map)

        return score_map, bbox, offset_map

    def cal_bbox(self, score_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map.flatten(1), dim=1, keepdim=True)
        idx_y = torch.div(idx, self.feat_sz, rounding_mode='floor')
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 4, 1) # torch.Size([32, 4, 1])
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        # offset: (l,t,r,b)
        # x1, y1, x2, y2
        # the offset should not divide the self.feat_sz, since I use the sigmoid to limit it in (0,1)
        bbox = torch.cat([idx_x.to(torch.float) / self.feat_sz - offset[:, :1],
                          idx_y.to(torch.float) / self.feat_sz - offset[:, 1:2],
                          idx_x.to(torch.float) / self.feat_sz + offset[:, 2:3],
                          idx_y.to(torch.float) / self.feat_sz + offset[:, 3:4],
                          ], dim=1)
        bbox = box_xyxy_to_cxcywh(bbox)
        if return_score:
            return bbox, max_score
        return bbox

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        x_cls = x
        for i, layer in enumerate(self.layers_cls):
            x_cls = F.relu(layer(x_cls)) if i < self.num_layers - 1 else layer(x_cls)
        x_cls = x_cls.permute(0,2,1).reshape(-1,1,self.feat_sz,self.feat_sz)

        x_reg = x
        for i, layer in enumerate(self.layers_reg):
            x_reg = F.relu(layer(x_reg)) if i < self.num_layers - 1 else layer(x_reg)
        x_reg = x_reg.permute(0, 2, 1).reshape(-1, 4, self.feat_sz, self.feat_sz)

        return _sigmoid(x_cls), _sigmoid(x_reg)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_head(cfg, encoder):
    num_channels_enc = encoder.num_channels
    stride = cfg.MODEL.ENCODER.STRIDE
    if cfg.MODEL.HEAD.TYPE == "MLP":
        in_channel = num_channels_enc
        hidden_dim = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        mlp_head = MLPPredictor(inplanes=in_channel, channel=hidden_dim,
                                feat_sz=feat_sz, stride=stride)
        return mlp_head
    elif "CORNER" in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = num_channels_enc
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        conv_type = cfg.MODEL.HEAD.CONV_TYPE
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride,
                                      conv_type=conv_type,
                                      xavier_init=cfg.MODEL.HEAD.XAVIER_INIT)
        return center_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD.TYPE)
