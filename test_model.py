"""
Eğitilmiş modeli otomatik test et
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from pathlib import Path
from nanochat.gpt_cpu import CPUCompatibleGPT
from nanochat.tokenizer import get_tokenizer

print("=" * 80)
print("🎭 MODEL TEST - Eğitilmiş Medium Modelini Test Et")
print("=" * 80)

# Model yolu
model_path = Path.home() / ".cache" / "nanochat" / "shakespeare" / "model_medium.pt"

if not model_path.exists():
    print(f"\n❌ Model bulunamadı: {model_path}")
    print("Önce eğitim yapın: python train_medium.py")
    exit(1)

print(f"\n📂 Model yükleniyor: {model_path}")

# Model yükle
checkpoint = torch.load(model_path, weights_only=False)

# Config'i al
config = checkpoint['config']
print(f"   ✅ Config: {config.n_layer} layers, {config.n_embd} dim")

# Model oluştur
model = CPUCompatibleGPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   ✅ Model yüklendi: {sum(p.numel() for p in model.parameters()):,} parametre")
print(f"   ✅ Step: {checkpoint['step']}, Val Loss: {checkpoint['val_loss']:.4f}")

# Tokenizer
tokenizer = get_tokenizer()
print(f"   ✅ Tokenizer hazır")

# Test promptları
prompts = [
    "ROMEO:",
    "JULIET:",
    "To be or not to be,",
    "What light through yonder window",
    "O Romeo, Romeo!",
    "The course of true love",
]

print("\n" + "=" * 80)
print("🎨 METİN ÜRETME TESTLERİ")
print("=" * 80)

for prompt in prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    print("-" * 80)
    
    # Tokenize ve tensor'e çevir
    tokens = tokenizer.encode(prompt)
    context = torch.tensor([tokens], dtype=torch.long)
    
    # Generate (manuel loop - eğitim scriptindeki gibi)
    with torch.no_grad():
        for _ in range(100):
            # Sequence length limitine göre kırp
            context_crop = context if context.size(1) <= config.sequence_len else context[:, -config.sequence_len:]
            logits, _ = model(context_crop)
            logits = logits[:, -1, :] / 0.8  # temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Top-k sampling
            if 200:
                top_k_probs, top_k_indices = torch.topk(probs, 200, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token = torch.gather(top_k_indices, -1, next_token_idx)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            context = torch.cat([context, next_token], dim=1)
    
    # Decode
    generated = tokenizer.decode(context[0].tolist())
    print(generated)
    print()

print("=" * 80)
print("✅ TEST TAMAMLANDI!")
print("=" * 80)
print("\n💡 İnteraktif mod için: python inference_cpu.py")
