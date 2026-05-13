import torch
import clip
import torch.nn as nn
from timm.models.layers import trunc_normal_

class TextEncoder(nn.Module):
    def __init__(self, type, slicing_mode, out_channel):
        super().__init__()
        # Load CLIP on CPU first; the whole model is moved to the target device
        # later via .to(device) in the tracker, which also moves CLIP parameters.
        import os
        device = os.environ.get('MVLM_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.clip, self.preprocess = clip.load(type, device=device)
        self.slicing_mode = slicing_mode
        clip_embed_dim = self.clip.text_projection.size(1)
        self.text_proj = nn.Linear(clip_embed_dim, out_channel)
        trunc_normal_(self.text_proj.weight, std=.02)
        nn.init.constant_(self.text_proj.bias, 0)

    @property
    def dtype(self):
        return self.text_proj.weight.dtype

    def slice_token(self, x, text_token):
        if self.slicing_mode == "full":
            sliced_token_emb = x
        elif self.slicing_mode == "eos_only":
            eos_index = text_token.argmax(dim=-1)
            sliced_token_emb = x[torch.arange(x.shape[0]), eos_index].unsqueeze(1)
        elif self.slicing_mode == "wo_pad":
            eos_index = text_token.argmax(dim=-1)
            max_seq_len = (eos_index + 1).max().item()
            sliced_token_emb = x[:, :max_seq_len, :]
        else:
            raise ValueError(f"Invalid slicing mode: {self.slicing_mode}")
        
        return sliced_token_emb

    def forward(self, raw_text):
        # raw_text (list) len: B
        text_token = self.clip_tokenizer(raw_text).to(self.text_proj.weight.device) # [B] -> [B, 77]

        if len(text_token.shape) == 1: # when inference, [77] -> [1, 77], ensuring [B, 77]
            text_token = text_token.unsqueeze(0)

        with torch.no_grad():
            x = self.clip.token_embedding(text_token).type(self.clip.dtype)  # [B, 77] -> [B, 77, clip_embed_dim]
            x = x + self.clip.positional_embedding.type(self.clip.dtype) # [B, 77, clip_embed_dim]
            x = x.permute(1, 0, 2)  # [B, 77, clip_embed_dim] -> [77, B, clip_embed_dim]
            x = self.clip.transformer(x) # [77, B, clip_embed_dim]
            x = x.permute(1, 0, 2) # [B, 77, clip_embed_dim]
            text_token_emb = self.slice_token(x, text_token)

        text_emb = self.text_proj(text_token_emb.type(self.dtype)) # [B, 77, 512]

        return text_emb

    def clip_tokenizer(self, raw_text):
        if raw_text is None:
            text_token = torch.zeros(77, dtype=torch.long)
        else:
            text_token = clip.tokenize(raw_text).squeeze(0)

        return text_token

def build_clip_textencoder(cfg, encoder):
    num_channels_enc = encoder.num_channels
    model = TextEncoder(cfg.MODEL.TEXT_ENCODER.TYPE, cfg.MODEL.TEXT_ENCODER.SLICING_MODE, num_channels_enc)
    return model

