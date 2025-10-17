"""
CPU-optimized GPT wrapper for Windows
BFloat16 kontrolleri devre dışı, float32 kullanır
"""

import torch
import torch.nn as nn
from nanochat.gpt import GPT as OriginalGPT, GPTConfig

class CPUCompatibleGPT(OriginalGPT):
    """CPU'da çalışabilen GPT versiyonu"""
    
    def __init__(self, config):
        super().__init__(config)
        # CPU için float32'ye dönüştür
        self.to(torch.float32)
        
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """CPU uyumlu rotary embeddings - float32 kullanır"""
        if device is None:
            device = self.transformer.wte.weight.device
        
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        
        # CPU için float32 kullan (bfloat16 yerine)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """CPU uyumlu forward pass - bfloat16 kontrolü yok"""
        B, T = idx.size()

        # Rotary embeddings kontrolü (bfloat16 assertion'ı kaldırıldı)
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        
        # bfloat16 kontrolü kaldırıldı - CPU float32 kullanıyor
        
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = self.norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = self.norm(x)

        # Forward the lm_head
        softcap = 15
        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            logits = torch.tanh(logits / softcap) * softcap
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                reduction=loss_reduction
            )
            return logits, loss
        else:
            logits = self.lm_head(x[:, -1:, :])
            logits = logits.float()
            logits = torch.tanh(logits / softcap) * softcap
            return logits, None
    
    def norm(self, x):
        """RMS normalization"""
        import torch.nn.functional as F
        return F.rms_norm(x, (x.size(-1),))
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """CPU için basitleştirilmiş optimizer - Muon yerine AdamW"""
        # CPU'da Muon yerine tüm parametreler için AdamW kullan
        all_params = list(self.parameters())
        
        optimizer = torch.optim.AdamW(
            all_params,
            lr=embedding_lr * 0.1,  # Daha düşük learning rate
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        return [optimizer]
