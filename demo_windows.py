"""
Windows için basit demo scripti
Küçük bir model ile tokenizer ve temel özellikleri test eder
"""

import os
import sys

# CUDA kullanmayı devre dışı bırak (CPU kullan)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("=" * 80)
print("🚀 nanochat Windows Demo")
print("=" * 80)

# 1. Tokenizer test
print("\n1️⃣ Tokenizer testi...")
try:
    from nanochat.tokenizer import get_tokenizer
    
    # Basit bir test metni
    test_text = "Hello, this is a test of the nanochat tokenizer!"
    print(f"   Test metni: '{test_text}'")
    
    # Not: Tokenizer henüz eğitilmediği için bu hata verebilir
    try:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   ✅ Tokenize edildi: {len(tokens)} token")
        print(f"   ✅ Decode edildi: '{decoded}'")
    except FileNotFoundError:
        print("   ⚠️  Tokenizer henüz eğitilmemiş. Önce tokenizer eğitmeniz gerekiyor.")
        print("   📝 Komut: python -m scripts.tok_train --max_chars=100000000")
        
except Exception as e:
    print(f"   ❌ Hata: {e}")

# 2. Model yapılandırma testi
print("\n2️⃣ Model yapılandırma testi...")
try:
    from nanochat.gpt import GPT, GPTConfig
    
    # Çok küçük bir model yapılandırması (CPU için)
    config = GPTConfig(
        vocab_size=50000,
        depth=4,  # Çok küçük - sadece test için
        max_seq_len=256,
        model_dim=256,
        num_heads=4,
        num_kv_heads=4,
    )
    
    print(f"   ✅ Model yapılandırması oluşturuldu:")
    print(f"      - Derinlik (depth): {config.depth}")
    print(f"      - Model boyutu: {config.model_dim}")
    print(f"      - Attention heads: {config.num_heads}")
    print(f"      - Max sequence length: {config.max_seq_len}")
    
    # Model oluştur (CPU'da)
    model = GPT(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model oluşturuldu: {num_params:,} parametre")
    
except Exception as e:
    print(f"   ❌ Hata: {e}")

# 3. PyTorch ve bağımlılıklar testi
print("\n3️⃣ Bağımlılıklar testi...")
try:
    import torch
    print(f"   ✅ PyTorch versiyon: {torch.__version__}")
    print(f"   ✅ CUDA mevcut: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA versiyon: {torch.version.cuda}")
        print(f"   ✅ GPU sayısı: {torch.cuda.device_count()}")
    else:
        print(f"   ℹ️  CPU modunda çalışıyor")
        
    import tiktoken
    print(f"   ✅ tiktoken yüklü")
    
    import fastapi
    print(f"   ✅ FastAPI yüklü")
    
except Exception as e:
    print(f"   ❌ Hata: {e}")

print("\n" + "=" * 80)
print("📊 Özet:")
print("=" * 80)
print("""
✅ Kurulum tamamlandı!

📝 Sonraki adımlar:

1. VERİ İNDİR (küçük örnek):
   python -m nanochat.dataset -n 1

2. TOKENIZER EĞİT:
   python -m scripts.tok_train --max_chars=100000000

3. MİNİ MODEL EĞİT (CPU - ÇOK YAVAŞ):
   python mini_train_cpu.py

4. VEYA BULUT GPU KULLAN:
   - Lambda Cloud'dan 8xH100 kirala
   - speedrun.sh scriptini çalıştır

ℹ️  Not: Ciddi model eğitimi için GPU şarttır!
   CPU'da sadece test ve demo yapılabilir.
""")
print("=" * 80)
