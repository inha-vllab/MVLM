import torch
import torch.nn as nn
import math

def clip_box_to_grid(box: torch.Tensor, H: int, W: int):
    # box: (B, 4) (x1,y1,w,h)
    box_clipped = torch.zeros_like(box)
    for b in range(box.shape[0]):
        x1, y1, w, h = box[b,:].unbind(-1)
        x1 = torch.max(torch.tensor(0.0, device=box.device), torch.min(torch.tensor(W, device=box.device), x1))
        y1 = torch.max(torch.tensor(0.0, device=box.device), torch.min(torch.tensor(H, device=box.device), y1))
        x2 = torch.min(torch.tensor(W, device=box.device), x1 + w)
        y2 = torch.min(torch.tensor(H, device=box.device), y1 + h)
        w = torch.max(torch.tensor(0.0, device=box.device), x2 - x1)
        h = torch.max(torch.tensor(0.0, device=box.device), y2 - y1)
        box_clipped[b,:] = torch.stack([x1, y1, w, h], dim=-1) # (B, 4) (x1,y1,w,h)
    return box_clipped

def px_to_tok_box(px_box: torch.Tensor, IMG_H, IMG_W, TOK_W, TOK_H):
    # (B, 4) -> (B, 4)
    x1, y1, w, h = px_box.unbind(-1)
    x2 = x1 + w
    y2 = y1 + h
    tx1 = (x1 / IMG_W * TOK_W).int()
    ty1 = (y1 / IMG_H * TOK_H).int()
    tx2 = (x2 / IMG_W * TOK_W).int()
    ty2 = (y2 / IMG_H * TOK_H).int()
    tw = (tx2 - tx1).int()
    th = (ty2 - ty1).int()
    return torch.stack([tx1, ty1, tw, th], dim=-1) # (B, 4) (x1,y1,w,h)

def extract_token_mask_from_bbox(bbox_px: torch.Tensor, IMG_H, IMG_W, TOK_H=14, TOK_W=14):
    """
    Extract token mask for the corresponding region in token map based on arbitrary bbox

    Args:
        bbox_px: (B, 4) (x1,y1,w,h)
        IMG_W, IMG_H: Image size (pixels)
        TOK_W, TOK_H: Token grid size

    Returns:
        token_mask: (B, TOK_H * TOK_W)
    """
    # Convert pixel coordinates to token coordinates
    tok_box = px_to_tok_box(bbox_px, IMG_H, IMG_W, TOK_H, TOK_W) # (B, 4) (x1,y1,w,h)
    tx1, ty1, tw, th = tok_box.unbind(-1)
    tx2 = tx1 + tw
    ty2 = ty1 + th
    
    token_mask = torch.zeros(bbox_px.shape[0], TOK_H * TOK_W, dtype=torch.bool, device=bbox_px.device)
    for b in range(bbox_px.shape[0]):
        for y in range(ty1[b].item(), ty2[b].item()):
            for x in range(tx1[b].item(), tx2[b].item()):
                token_mask[b, y * TOK_W + x] = True

    return token_mask # (B, TOK_H * TOK_W)

def get_vl_sim(tokens: torch.Tensor, mean_lang_token: torch.Tensor):
    # tokens: (B, N, D)
    # mean_lang_token: (B, 1, D)
    return torch.matmul(tokens, mean_lang_token.transpose(-1, -2)) # (B, N)

def get_max_vlsim_token(tokens: torch.Tensor, mean_lang_token: torch.Tensor):
    # tokens: (N, D)
    # mean_lang_token: (1, D)
    max_vlsim_score = torch.max(get_vl_sim(tokens, mean_lang_token)).unsqueeze(0) # (1, 1)
    return tokens[torch.argmax(get_vl_sim(tokens, mean_lang_token))].unsqueeze(0), max_vlsim_score # (1, D), (1, 1)

class CMloss(nn.Module):
    """
    Contrastive Margin (CM) Loss with log-sum-exp formulation.
    Maximizes margin Δρ = ρ(pos) - ρ(neg) via minimizing exp(-c·Δρ·|Δρ|).
    
    Based on Theorem 1 (mis-localization bound):
        L = log(Σ exp(-c·Δρ·|Δρ|))
    where Δρ = ρ(GT) - ρ(neg) is the vision-language correlation margin.
    """
    def __init__(self, temperature=1.0, top_k=10):
        super().__init__()
        self.temp = temperature
        self.top_k = top_k
    
    def forward(self, lang_token_map, vis_token_map, gt_bbox, IMG_H, IMG_W):
        """
        Args:
            lang_token_map: (B, L, D) language token embeddings
            vis_token_map: (B, N_tok, D) vision token embeddings
            gt_bbox: (B, 4) ground truth boxes in (x,y,w,h) format
            IMG_H, IMG_W: image dimensions in pixels
        Returns:
            scalar loss encouraging large margins between GT and background
        """
        # Normalize embeddings
        mean_lang_token = lang_token_map.mean(dim=1, keepdim=True)  # (B, 1, D)
        mean_lang_token = mean_lang_token / mean_lang_token.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        vis_token_map = vis_token_map / vis_token_map.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, N_tok, D)
        
        B = gt_bbox.shape[0]
        TOK_H = TOK_W = int(math.sqrt(vis_token_map.shape[1]))
        
        # Clip bbox to valid range
        gt_bbox = clip_box_to_grid(gt_bbox, IMG_H, IMG_W)
        
        # Extract GT token mask
        gt_token_mask = extract_token_mask_from_bbox(gt_bbox, IMG_H, IMG_W, TOK_H, TOK_W)  # (B, N_tok)
        
        losses = []
        for b in range(B):
            # GT tokens (positive)
            gt_tokens = vis_token_map[b][gt_token_mask[b]]  # (N_gt, D)
            if gt_tokens.shape[0] == 0:
                continue
            
            # Get max VL-sim token as positive representative
            vl_sim_gt = torch.matmul(gt_tokens, mean_lang_token[b].transpose(-1, -2)).squeeze(-1)  # (N_gt,)
            rho_pos = vl_sim_gt.mean()  # scalar - best GT token correlation
            
            # Negative tokens (background)
            neg_tokens = vis_token_map[b][~gt_token_mask[b]]  # (N_neg, D)
            if neg_tokens.shape[0] == 0:
                # Skip if no negative tokens (degenerate case)
                continue
            
            vl_sim_neg = torch.matmul(neg_tokens, mean_lang_token[b].transpose(-1, -2)).squeeze(-1)  # (N_neg,)
            
            # Margin: Δρ = ρ(pos) - ρ(neg)
            # Larger margin = better separation between GT and background
            margins = rho_pos - vl_sim_neg  # (N_neg,)
            
            # Theory: minimize Σ exp(-c·Δρ²) with signed handling
            # Positive margin (Δρ > 0): exp(-c·Δρ·|Δρ|) < 1 (low loss, good)
            # Negative margin (Δρ < 0): exp(+c·Δρ·|Δρ|) > 1 (high loss, penalty!)
            # Using: -c · Δρ · |Δρ| to preserve sign while squaring magnitude
            exponents = -self.temp * margins * margins.abs()  # (N_neg,)
            
            # Top-K hard negatives (largest exponents = worst cases)
            # Now includes both small positive margins and negative margins
            if self.top_k > 0 and exponents.shape[0] > self.top_k:
                # Select largest exponents = hardest negatives (small/negative margins)
                exponents, _ = exponents.topk(self.top_k, largest=True)
            
            losses.append(torch.exp(exponents).mean())
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=vis_token_map.device, requires_grad=True)
        
        return torch.stack(losses).mean()