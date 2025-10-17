"""
HIZLI TEST - 5-10 dakika
Bilgisayarınız için optimize edilmiş hızlı test
Sistem: 12 core CPU, 15.7 GB RAM
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
print("⚡ HIZLI TEST - Bilgisayarınıza Özel")
print("=" * 80)

# Veri indirme
print("\n📥 Shakespeare veri seti indiriliyor...")
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

print("\n🔤 Tokenizer yükleniyor...")
tokenizer = get_tokenizer()
tokens = tokenizer.encode(text)

split_idx = int(len(tokens) * 0.9)
train_tokens = torch.tensor(tokens[:split_idx], dtype=torch.long)
val_tokens = torch.tensor(tokens[split_idx:], dtype=torch.long)
print(f"   ✅ Train: {len(train_tokens):,} tokens")
print(f"   ✅ Val: {len(val_tokens):,} tokens")

# HIZLI TEST YAPIŞANDIRMASI
print("\n🏗️  Model oluşturuluyor (HIZLI TEST)...")
config = GPTConfig(
    sequence_len=128,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=2,           # ⚡ Çok küçük
    n_head=2,
    n_kv_head=2,
    n_embd=128,          # ⚡ Çok küçük
)

device = "cpu"
model = CPUCompatibleGPT(config)
num_params = sum(p.numel() for p in model.parameters())
print(f"   ✅ Model: {num_params:,} parametre (~17M)")
print(f"   ✅ RAM tahmini: ~300 MB")

optimizers = model.setup_optimizers(embedding_lr=0.001, unembedding_lr=0.001, matrix_lr=0.001)
optimizer = optimizers[0]

# Eğitim parametreleri
batch_size = 4
seq_len = 128
max_steps = 50          # ⚡ Hızlı test
eval_interval = 10
print_interval = 5

print(f"\n🏋️  Eğitim Parametreleri:")
print(f"   - Max steps: {max_steps}")
print(f"   - Batch size: {batch_size}")
print(f"   - Tahmini süre: 5-10 dakika")

def get_batch(split):
    data = train_tokens if split == 'train' else val_tokens
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split in ['train', 'val']:
        loss_sum = 0.0
        for _ in range(5):
            x, y = get_batch(split)
            _, loss = model(x, y)
            loss_sum += loss.item()
        losses[split] = loss_sum / 5
    model.train()
    return losses

print("\n" + "=" * 80)
print("🚀 EĞİTİM BAŞLIYOR (HIZLI TEST)")
print("=" * 80)

model.train()
best_val_loss = float('inf')
start_time = time.time()

for step in range(max_steps):
    step_start = time.time()
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step_time = time.time() - step_start
    
    if step % print_interval == 0:
        print(f"Step {step:3d}/{max_steps} | Loss: {loss.item():.4f} | {step_time:.2f}s/step")
    
    if step % eval_interval == 0 and step > 0:
        losses = estimate_loss()
        print(f"\n📊 @ step {step}: Train={losses['train']:.4f}, Val={losses['val']:.4f}\n")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint_path = data_dir / "model_fast.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"   💾 Model kaydedildi\n")

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("✅ HIZLI TEST TAMAMLANDI!")
print("=" * 80)
print(f"Süre: {total_time/60:.1f} dakika")
print(f"En iyi val loss: {best_val_loss:.4f}")

# Basit inference
print("\n🎭 Metin üretme testi:")
model.eval()
prompt = "ROMEO:"
context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    for _ in range(50):
        context_crop = context if context.size(1) <= config.sequence_len else context[:, -config.sequence_len:]
        logits, _ = model(context_crop)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

generated = tokenizer.decode(context[0].tolist())
print(f"\n{generated}\n")

print("=" * 80)
print("✅ Test başarılı! Daha iyi sonuçlar için:")
print("   python train_medium.py  (30-60 dakika)")
print("   python train_max.py     (3-5 saat)")
print("=" * 80)
