"""
MAKSİMUM - 3-5 saat
Bilgisayarınızın maksimum kapasitesini kullanan eğitim
Sistem: 12 core CPU, 15.7 GB RAM
⚠️  Eğitim sırasında diğer programları kapatın!
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import torch
import requests
from pathlib import Path
from dataclasses import asdict

from nanochat.gpt_cpu import CPUCompatibleGPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

print("=" * 80)
print("🚀 MAKSİMUM KAPASİTE - Bilgisayarınıza Özel")
print("=" * 80)
print("⚠️  DİKKAT: 3-5 saat sürecek!")
print("⚠️  Eğitim sırasında diğer programları kapatın!")
print("=" * 80)

# Veri indirme
print("\n📥 Shakespeare veri seti indiriliyor...")
data_dir = Path.home() / ".cache" / "nanochat" / "shakespeare"
data_dir.mkdir(parents=True, exist_ok=True)
data_file = data_dir / "shakespeare.txt"

if not data_file.exists():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
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

# MAKSİMUM YAPILANDIRMA
print("\n🏗️  Model oluşturuluyor (MAKSİMUM)...")
config = GPTConfig(
    sequence_len=128,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=8,           # 🚀 Büyük
    n_head=8,
    n_kv_head=8,
    n_embd=512,          # 🚀 Büyük
)

device = "cpu"
model = CPUCompatibleGPT(config)
num_params = sum(p.numel() for p in model.parameters())
print(f"   ✅ Model: {num_params:,} parametre (~92M)")
print(f"   ✅ RAM tahmini: ~4-5 GB")

optimizers = model.setup_optimizers(embedding_lr=0.0008, unembedding_lr=0.0008, matrix_lr=0.0008)
optimizer = optimizers[0]

# Eğitim parametreleri
batch_size = 4
seq_len = 128
max_steps = 500         # 🚀 Uzun eğitim
eval_interval = 25
print_interval = 10

print(f"\n🏋️  Eğitim Parametreleri:")
print(f"   - Max steps: {max_steps}")
print(f"   - Batch size: {batch_size}")
print(f"   - Model parametreleri: {num_params:,}")
print(f"   - Tahmini süre: 3-5 saat")
print(f"\n⚠️  RAM kullanımını izleyin!")

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
print("🚀 EĞİTİM BAŞLIYOR (MAKSİMUM)")
print("=" * 80)
print("⏰ 3-5 saat sürecek - Uzun eğitim!")
print("💡 İpucu: Bilgisayarı başka işler için kullanmayın\n")

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
        remaining = (step_time * (max_steps - step)) / 3600  # saat
        progress = (step / max_steps) * 100
        print(f"Step {step:3d}/{max_steps} ({progress:.1f}%) | Loss: {loss.item():.4f} | {step_time:.2f}s | Kalan: ~{remaining:.1f}saat")
    
    if step % eval_interval == 0 and step > 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"\n📊 @ step {step} ({elapsed/3600:.1f} saat):")
        print(f"   Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            improvement = "🎯 YENİ EN İYİ!"
            checkpoint_path = data_dir / "model_max.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': asdict(config),
                'num_params': num_params,
            }, checkpoint_path)
            print(f"   💾 En iyi model kaydedildi: {checkpoint_path}")
        else:
            improvement = ""
        print(f"   En iyi val: {best_val_loss:.4f} {improvement}\n")
    
    # Her 100 adımda bir ara checkpoint
    if step % 100 == 0 and step > 0:
        temp_path = data_dir / f"checkpoint_step_{step}.pt"
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': asdict(config),
        }, temp_path)

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("✅ EĞİTİM TAMAMLANDI!")
print("=" * 80)
print(f"Toplam süre: {total_time/3600:.1f} saat ({total_time/60:.0f} dakika)")
print(f"En iyi val loss: {best_val_loss:.4f}")
print(f"Model: {data_dir / 'model_max.pt'}")
print(f"Parametre sayısı: {num_params:,}")

# Detaylı inference testi
print("\n" + "=" * 80)
print("🎭 Metin Üretme Testi")
print("=" * 80)

model.eval()

test_prompts = [
    "ROMEO:",
    "JULIET:",
    "To be or not to be,",
    "Friends, Romans, countrymen,",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: '{prompt}'")
    print('='*60)
    
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(150):  # Daha uzun metin
            context_crop = context if context.size(1) <= config.sequence_len else context[:, -config.sequence_len:]
            logits, _ = model(context_crop)
            logits = logits[:, -1, :] / 0.8  # temperature
            
            # Top-k sampling
            top_k = 40
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
    
    generated = tokenizer.decode(context[0].tolist())
    print(generated)

print("\n" + "=" * 80)
print("🎉 MAKSİMUM KAPASİTE EĞİTİMİ BAŞARILI!")
print("=" * 80)
print(f"""
✅ {num_params:,} parametreli büyük model eğitildi
✅ Eğitim süresi: {total_time/3600:.1f} saat
✅ En iyi val loss: {best_val_loss:.4f}
✅ Model boyutu: ~{num_params * 4 / (1024**2):.0f} MB

📊 Model Detayları:
   - Layers: {config.n_layer}
   - Embedding dim: {config.n_embd}
   - Attention heads: {config.n_head}

🎮 Modelinizi kullanın:
   Önce inference_cpu.py'yi güncelleyin:
   - checkpoint_path = data_dir / "model_max.pt"
   
   Sonra çalıştırın:
   python inference_cpu.py

💡 Bu bilgisayarınızda CPU ile eğitilebilecek en büyük model!
   Daha büyük modeller için GPU gerekir.

🏆 Tebrikler! Maksimum kapasiteyi kullandınız! 🚀
""")
print("=" * 80)
