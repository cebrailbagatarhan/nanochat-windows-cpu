"""
Windows CPU için Basitleştirilmiş nanochat Eğitimi
Çok küçük bir model ile Shakespeare verisi üzerinde eğitim

UYARI: Bu CPU'da yavaş olacak ama çalışacak!
"""

import os
import time
import torch
import requests
from pathlib import Path
from dataclasses import asdict

# CPU kullanımını zorla
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from nanochat.gpt_cpu import CPUCompatibleGPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

print("=" * 80)
print("🎭 nanochat CPU Eğitimi - Shakespeare Örneği")
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
    print(f"   ✅ İndirildi: {len(response.text):,} karakter")
else:
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"   ✅ Zaten mevcut: {len(text):,} karakter")

# Veriyi oku
with open(data_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer kullan
print("\n🔤 Tokenizer yükleniyor...")
tokenizer = get_tokenizer()
print(f"   ✅ Vocab size: {tokenizer.get_vocab_size():,}")

# Metni tokenize et
print("\n⚙️  Metin tokenize ediliyor...")
tokens = tokenizer.encode(text)
print(f"   ✅ Token sayısı: {len(tokens):,}")

# Train/val split
split_idx = int(len(tokens) * 0.9)
train_tokens = torch.tensor(tokens[:split_idx], dtype=torch.long)
val_tokens = torch.tensor(tokens[split_idx:], dtype=torch.long)
print(f"   ✅ Train: {len(train_tokens):,} tokens")
print(f"   ✅ Val: {len(val_tokens):,} tokens")

# Orta seviye model yapılandırması (Bilgisayarınıza göre optimize)
# Sistem: 12 CPU cores, 15.7 GB RAM
print("\n🏗️  Model oluşturuluyor...")
config = GPTConfig(
    sequence_len=128,        # Kısa context (normalde 2048)
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=6,               # Orta seviye (normalde 20)
    n_head=6,                # Dengeli (normalde 10)
    n_kv_head=6,
    n_embd=384,              # Orta boyut (normalde 1280)
)

print(f"   📐 Yapılandırma:")
print(f"      - Katmanlar: {config.n_layer}")
print(f"      - Embedding: {config.n_embd}")
print(f"      - Heads: {config.n_head}")
print(f"      - Context: {config.sequence_len}")

device = "cpu"
model = CPUCompatibleGPT(config)
num_params = sum(p.numel() for p in model.parameters())
print(f"   ✅ Model oluşturuldu: {num_params:,} parametre")
print(f"   ✅ Device: {device}")

# Optimizer
print("\n🎯 Optimizer ayarlanıyor...")
optimizers = model.setup_optimizers(
    embedding_lr=0.001,
    unembedding_lr=0.001,
    matrix_lr=0.001
)
optimizer = optimizers[0]
print(f"   ✅ AdamW optimizer hazır")

# Eğitim parametreleri (Bilgisayarınıza göre optimize)
batch_size = 4          # CPU için uygun batch
seq_len = 128           # Kısa sequence
max_steps = 200         # Dengeli eğitim (~30-60 dakika)
eval_interval = 20      # Her 20 adımda değerlendirme
print_interval = 10     # Her 10 adımda log

print(f"\n🏋️  Eğitim Parametreleri:")
print(f"   - Batch size: {batch_size}")
print(f"   - Sequence length: {seq_len}")
print(f"   - Max steps: {max_steps}")
print(f"   - Learning rate: {optimizer.param_groups[0]['lr']}")

def get_batch(split):
    """Rastgele bir batch al"""
    data = train_tokens if split == 'train' else val_tokens
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    """Loss'u değerlendir"""
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split in ['train', 'val']:
        loss_sum = 0.0
        for _ in range(10):  # 10 batch ortalaması
            x, y = get_batch(split)
            _, loss = model(x, y)
            loss_sum += loss.item()
        losses[split] = loss_sum / 10
    model.train()
    return losses

# Eğitim loop
print("\n" + "=" * 80)
print("🚀 EĞİTİM BAŞLIYOR")
print("=" * 80)
print("⏰ CPU'da yavaş olacak - sabırlı olun!\n")

model.train()
best_val_loss = float('inf')
start_time = time.time()

for step in range(max_steps):
    step_start = time.time()
    
    # Batch al
    x, y = get_batch('train')
    
    # Forward pass
    logits, loss = model(x, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    step_time = time.time() - step_start
    
    # Log
    if step % print_interval == 0:
        print(f"Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | {step_time:.2f}s/step")
    
    # Evaluation
    if step % eval_interval == 0 and step > 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"\n📊 Değerlendirme @ step {step}:")
        print(f"   Train loss: {losses['train']:.4f}")
        print(f"   Val loss:   {losses['val']:.4f}")
        print(f"   Geçen süre: {elapsed/60:.1f} dakika\n")
        
        # En iyi modeli kaydet
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint_path = data_dir / "best_model.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': asdict(config),
            }, checkpoint_path)
            print(f"   💾 En iyi model kaydedildi: {checkpoint_path}")

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("✅ EĞİTİM TAMAMLANDI!")
print("=" * 80)
print(f"Toplam süre: {total_time/60:.1f} dakika")
print(f"En iyi val loss: {best_val_loss:.4f}")
print(f"Model: {data_dir / 'best_model.pt'}")

# Basit bir inference testi
print("\n" + "=" * 80)
print("🎭 Metin Üretme Testi")
print("=" * 80)

model.eval()
prompt = "ROMEO:"
print(f"Prompt: '{prompt}'")
print(f"Üretilen metin:\n")

# Basit greedy sampling
context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
max_new_tokens = 100

with torch.no_grad():
    for _ in range(max_new_tokens):
        # Context'i kırp
        context_crop = context if context.size(1) <= config.sequence_len else context[:, -config.sequence_len:]
        # Predict
        logits, _ = model(context_crop)
        logits = logits[:, -1, :]  # Son token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

generated = tokenizer.decode(context[0].tolist())
print(generated)

print("\n" + "=" * 80)
print("🎉 BAŞARILI!")
print("=" * 80)
print(f"""
✅ Küçük bir GPT modeli CPU'da eğitildi
✅ Shakespeare metinleri üzerinde öğrendi
✅ Metin üretebiliyor (kalite düşük ama çalışıyor!)

📝 Model bilgileri:
   - Parametre sayısı: {num_params:,}
   - Eğitim süresi: {total_time/60:.1f} dakika
   - En iyi val loss: {best_val_loss:.4f}
   - Kayıt yeri: {data_dir / 'best_model.pt'}

💡 Daha iyi sonuçlar için:
   - max_steps değerini artırın (örn. 1000)
   - Model boyutunu büyütün (n_layer, n_embd)
   - Daha fazla veri kullanın
   - VE EN ÖNEMLİSİ: GPU kullanın! 🚀
""")
