"""
Hızlı Test - Sadece 10 adım (2-3 dakika)
Sistemin çalıştığını doğrulamak için
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import torch
import requests
from pathlib import Path

from nanochat.gpt_cpu import CPUCompatibleGPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

print("=" * 80)
print("⚡ HIZLI TEST - 10 Adım Eğitim")
print("=" * 80)

# Küçük veri indir
print("\n📥 Veri indiriliyor...")
data_dir = Path.home() / ".cache" / "nanochat" / "shakespeare"
data_dir.mkdir(parents=True, exist_ok=True)
data_file = data_dir / "shakespeare.txt"

if not data_file.exists():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open(data_file, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"   ✅ İndirildi")
else:
    print(f"   ✅ Zaten mevcut")

with open(data_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer
tokenizer = get_tokenizer()
tokens = tokenizer.encode(text)
train_tokens = torch.tensor(tokens[:10000], dtype=torch.long)  # Sadece ilk 10k token

print(f"   ✅ {len(train_tokens):,} token hazır")

# Mini model
config = GPTConfig(
    sequence_len=64,     # Çok kısa
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=2,           # Çok küçük
    n_head=2,
    n_kv_head=2,
    n_embd=128,          # Çok küçük
)

print(f"\n🏗️  Model oluşturuluyor...")
model = CPUCompatibleGPT(config)
num_params = sum(p.numel() for p in model.parameters())
print(f"   ✅ {num_params:,} parametre")

optimizer = model.setup_optimizers()[0]

def get_batch():
    ix = torch.randint(len(train_tokens) - 64, (2,))
    x = torch.stack([train_tokens[i:i+64] for i in ix])
    y = torch.stack([train_tokens[i+1:i+65] for i in ix])
    return x, y

print(f"\n🏃 10 adım eğitim başlıyor...")
model.train()
start_time = time.time()

for step in range(10):
    step_start = time.time()
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - step_start
    print(f"   Adım {step+1}/10 | Loss: {loss.item():.4f} | {elapsed:.2f}s")

total_time = time.time() - start_time

print(f"\n✅ Tamamlandı! Toplam: {total_time:.1f} saniye")

# Hızlı inference testi
print(f"\n🎭 Metin üretme testi:")
model.eval()
prompt = "ROMEO:"
context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

with torch.no_grad():
    for _ in range(30):
        context_crop = context if context.size(1) <= 64 else context[:, -64:]
        logits, _ = model(context_crop)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

generated = tokenizer.decode(context[0].tolist())
print(f"\n{generated}\n")

print("=" * 80)
print("✅ SİSTEM ÇALIŞIYOR!")
print("=" * 80)
print(f"""
Sistem testi başarılı! nanochat CPU'da çalışıyor.

Şimdi şunları yapabilirsiniz:

1. Gerçek eğitim (15-30 dakika):
   python train_cpu_shakespeare.py

2. Daha uzun eğitim için train_cpu_shakespeare.py içinde:
   max_steps = 100 → 500 veya 1000

3. Daha güçlü model için:
   n_layer = 4 → 8
   n_embd = 256 → 512

İyi eğlenceler! 🚀
""")
