"""
ORTA SEVİYE - 30-60 dakika (ÖNERİLEN)
Bilgisayarınız için optimize edilmiş dengeli eğitim
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
print("⭐ ORTA SEVİYE - Bilgisayarınıza Özel (ÖNERİLEN)")
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

# ORTA SEVİYE YAPILANDIRMA
print("\n🏗️  Model oluşturuluyor (ORTA SEVİYE)...")
config = GPTConfig(
    sequence_len=128,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=6,           # ⭐ Dengeli
    n_head=6,
    n_kv_head=6,
    n_embd=384,          # ⭐ Dengeli
)

device = "cpu"
model = CPUCompatibleGPT(config)
num_params = sum(p.numel() for p in model.parameters())
print(f"   ✅ Model: {num_params:,} parametre (~61M)")
print(f"   ✅ RAM tahmini: ~2-3 GB")

optimizers = model.setup_optimizers(embedding_lr=0.001, unembedding_lr=0.001, matrix_lr=0.001)
optimizer = optimizers[0]

# Eğitim parametreleri
batch_size = 4
seq_len = 128
max_steps = 200         # ⭐ Dengeli
eval_interval = 20
print_interval = 10

print(f"\n🏋️  Eğitim Parametreleri:")
print(f"   - Max steps: {max_steps}")
print(f"   - Batch size: {batch_size}")
print(f"   - Tahmini süre: 30-60 dakika")

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
        for _ in range(10):
            x, y = get_batch(split)
            _, loss = model(x, y)
            loss_sum += loss.item()
        losses[split] = loss_sum / 10
    model.train()
    return losses

print("\n" + "=" * 80)
print("🚀 EĞİTİM BAŞLIYOR (ORTA SEVİYE)")
print("=" * 80)
print("⏰ 30-60 dakika sürecek - Sabırlı olun!\n")

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
        elapsed = time.time() - start_time
        remaining = (step_time * (max_steps - step)) / 60
        print(f"Step {step:3d}/{max_steps} | Loss: {loss.item():.4f} | {step_time:.2f}s | Kalan: ~{remaining:.0f}dk")
    
    if step % eval_interval == 0 and step > 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"\n📊 @ step {step} ({elapsed/60:.1f}dk):")
        print(f"   Train: {losses['train']:.4f} | Val: {losses['val']:.4f}\n")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint_path = data_dir / "model_medium.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"   💾 En iyi model kaydedildi: {checkpoint_path}\n")

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("✅ EĞİTİM TAMAMLANDI!")
print("=" * 80)
print(f"Toplam süre: {total_time/60:.1f} dakika")
print(f"En iyi val loss: {best_val_loss:.4f}")
print(f"Model: {data_dir / 'model_medium.pt'}")

# Inference testi
print("\n" + "=" * 80)
print("🎭 Metin Üretme Testi")
print("=" * 80)

model.eval()

test_prompts = [
    "ROMEO:",
    "JULIET:",
    "To be or not to be,",
]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(100):
            context_crop = context if context.size(1) <= config.sequence_len else context[:, -config.sequence_len:]
            logits, _ = model(context_crop)
            logits = logits[:, -1, :] / 0.8  # temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
    
    generated = tokenizer.decode(context[0].tolist())
    print(generated)

print("\n" + "=" * 80)
print("🎉 BAŞARILI!")
print("=" * 80)
print(f"""
✅ {num_params:,} parametreli model eğitildi
✅ Eğitim süresi: {total_time/60:.1f} dakika
✅ En iyi val loss: {best_val_loss:.4f}

🎮 Modelinizi kullanın:
   python inference_cpu.py

💡 Daha iyi sonuç için:
   python train_max.py (3-5 saat, daha güçlü model)
""")
print("=" * 80)
