"""
Windows CPU için minimal eğitim scripti
UYARI: Bu çok yavaş olacak! Sadece demo amaçlıdır.
Gerçek eğitim için GPU gereklidir.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU kullan

import torch
import time
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

print("=" * 80)
print("🐢 CPU Minimal Eğitim Demo (ÇOK YAVAŞ!)")
print("=" * 80)

# Çok küçük bir model yapılandırması
config = GPTConfig(
    sequence_len=128,      # Çok kısa (normalde 2048)
    vocab_size=65536,      # nanochat default
    n_layer=2,             # Sadece 2 katman (normalde 20)
    n_head=4,              # Az head (normalde 10)
    n_kv_head=4,
    n_embd=256,            # Küçük embedding (normalde 1280)
)

print(f"\n📐 Model Yapılandırması:")
print(f"   - Katman sayısı: {config.n_layer}")
print(f"   - Embedding boyutu: {config.n_embd}")
print(f"   - Sequence uzunluğu: {config.sequence_len}")

# Model oluştur
device = "cpu"
model = GPT(config)

# CPU için float32 kullan (bfloat16 desteklenmez)
model = model.to(device).to(torch.float32)

# Parametre sayısını hesapla
num_params = sum(p.numel() for p in model.parameters())
print(f"   - Toplam parametre: {num_params:,}")
print(f"   - Cihaz: {device}")

# Basit bir optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Dummy veri oluştur (gerçek veri yerine)
print(f"\n🎲 Rastgele eğitim verisi oluşturuluyor...")
batch_size = 2  # Çok küçük batch
seq_len = 128

# 10 adım için dummy veri
num_steps = 10
print(f"\n🏃 {num_steps} adım eğitim başlatılıyor...")
print("⚠️  Bu CPU'da çok yavaş olacak! Sabırlı olun...")

model.train()
for step in range(num_steps):
    start_time = time.time()
    
    # Rastgele veri oluştur
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass
    logits, loss = model(x, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    elapsed = time.time() - start_time
    
    print(f"   Adım {step+1}/{num_steps} | Loss: {loss.item():.4f} | Süre: {elapsed:.2f}s")

print("\n✅ Mini eğitim tamamlandı!")
print("\n" + "=" * 80)
print("📝 Sonuçlar:")
print("=" * 80)
print(f"""
✅ {num_steps} eğitim adımı tamamlandı
✅ Model çalışıyor (ama eğitilmedi)

⚠️  ÖNEMLİ NOTLAR:
1. Bu sadece kod çalışıyor mu diye test amaçlı
2. Gerçek veri kullanılmadı (rastgele sayılar)
3. CPU'da çok yavaş (GPU'da 1000x daha hızlı)
4. Bu kadar küçük model hiçbir şey öğrenemez

🚀 GERÇEKTEn KULLANMAK İÇİN:
1. Lambda Cloud'dan GPU kirala (8xH100)
2. speedrun.sh scriptini çalıştır
3. 4 saat bekle
4. $100'a gerçek ChatGPT'nin sahip ol!

💡 Alternatif: Eğitilmiş modeli indir
   Hugging Face'de başkaları eğitmiş olabilir:
   https://huggingface.co/models?search=nanochat
""")
print("=" * 80)
