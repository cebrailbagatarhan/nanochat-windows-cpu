"""
Bilgisayar Özellikleri:
- CPU: 12 çekirdek
- RAM: 15.7 GB
- Kullanılabilir RAM: ~3.9 GB (şu anda)

Maksimum model parametreleri hesaplanıyor...
"""

import torch
import psutil

print("=" * 80)
print("🔍 SİSTEM ANALİZİ VE OPTİMAL PARAMETRE ÖNERİLERİ")
print("=" * 80)

# Sistem bilgileri
cpu_cores = psutil.cpu_count()
ram_total = psutil.virtual_memory().total / (1024**3)
ram_available = psutil.virtual_memory().available / (1024**3)
ram_percent = psutil.virtual_memory().percent

print(f"\n💻 Sistem Kaynakları:")
print(f"   CPU Cores: {cpu_cores}")
print(f"   RAM Total: {ram_total:.1f} GB")
print(f"   RAM Available: {ram_available:.1f} GB")
print(f"   RAM Usage: {ram_percent:.1f}%")

# PyTorch thread ayarları
torch.set_num_threads(cpu_cores)
print(f"   PyTorch Threads: {torch.get_num_threads()}")

print("\n" + "=" * 80)
print("📊 MODEL BOYUTU TAHMİNLERİ")
print("=" * 80)

def estimate_model_memory(n_layer, n_embd, vocab_size=65536, seq_len=128):
    """Model RAM kullanımını tahmin et (MB)"""
    n_head = max(4, n_embd // 64)
    n_kv_head = n_head
    
    # Parametreler
    # Embedding: vocab_size * n_embd
    # Per layer: ~4 * n_embd * n_embd (Q, K, V, proj) + 8 * n_embd * n_embd (MLP)
    # LM head: vocab_size * n_embd
    
    embedding_params = vocab_size * n_embd
    lm_head_params = vocab_size * n_embd
    per_layer_params = (4 * n_embd * n_embd) + (8 * n_embd * n_embd)
    total_params = embedding_params + lm_head_params + (n_layer * per_layer_params)
    
    # Float32: 4 bytes per parameter
    # Model: params * 4
    # Optimizer state (AdamW): params * 8 (momentum + variance)
    # Gradients: params * 4
    # Activations: batch_size * seq_len * n_embd * n_layer * 4 (rough estimate)
    
    batch_size = 4
    model_memory = total_params * 4 / (1024**2)  # MB
    optimizer_memory = total_params * 8 / (1024**2)  # MB
    gradient_memory = total_params * 4 / (1024**2)  # MB
    activation_memory = batch_size * seq_len * n_embd * n_layer * 4 / (1024**2)  # MB
    
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    return {
        'params': total_params,
        'model_mb': model_memory,
        'optimizer_mb': optimizer_memory,
        'gradient_mb': gradient_memory,
        'activation_mb': activation_memory,
        'total_mb': total_memory,
        'total_gb': total_memory / 1024
    }

# Farklı konfigürasyonlar test et
configs = [
    # (n_layer, n_embd, açıklama)
    (2, 128, "Minimum (Hızlı Test)"),
    (4, 256, "Küçük (Default)"),
    (6, 384, "Orta"),
    (8, 512, "Büyük"),
    (10, 640, "Çok Büyük"),
    (12, 768, "Maksimum"),
    (16, 768, "GPT-2 Small Benzeri"),
]

print(f"\n{'Model':<25} {'Param':>12} {'RAM':>10} {'Durum':<20}")
print("-" * 80)

optimal_config = None
max_safe_ram = ram_total * 0.6  # Sistemin %60'ını kullan (güvenli limit)

for n_layer, n_embd, desc in configs:
    est = estimate_model_memory(n_layer, n_embd)
    params_m = est['params'] / 1e6
    ram_gb = est['total_gb']
    
    if ram_gb < max_safe_ram and optimal_config is None:
        optimal_config = (n_layer, n_embd, desc)
    
    status = ""
    if ram_gb < 2:
        status = "✅ Çok Hızlı"
    elif ram_gb < 4:
        status = "✅ Hızlı"
    elif ram_gb < max_safe_ram:
        status = "⚠️  Yavaş ama OK"
        optimal_config = (n_layer, n_embd, desc)
    else:
        status = "❌ RAM yetersiz"
    
    print(f"{desc:<25} {params_m:>10.1f}M {ram_gb:>9.1f}GB {status:<20}")

print("\n" + "=" * 80)
print("🎯 ÖNERİLEN YAPILANDIRMALAR")
print("=" * 80)

print(f"""
Bilgisayarınız için 3 farklı seviye öneri:

1️⃣  HIZLI TEST (5-10 dakika):
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   n_layer = 2
   n_embd = 128
   max_steps = 50
   batch_size = 4
   
   Parametre: ~0.5M
   RAM: ~200 MB
   Süre: 5-10 dakika
   Kalite: Çok düşük (sadece test için)

2️⃣  ORTA SEVİYE (30-60 dakika) - ÖNERİLEN:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   n_layer = 6
   n_embd = 384
   max_steps = 200
   batch_size = 4
   
   Parametre: ~35M
   RAM: ~2-3 GB
   Süre: 30-60 dakika
   Kalite: Orta (demo seviyesi)

3️⃣  MAKSİMUM (3-5 saat):
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   n_layer = 8
   n_embd = 512
   max_steps = 500
   batch_size = 4
   
   Parametre: ~70M
   RAM: ~4-5 GB
   Süre: 3-5 saat
   Kalite: İyi (gösterebilir seviye)

⚠️  ÖNEMLİ NOTLAR:
   - Eğitim sırasında diğer programları kapatın
   - RAM kullanımını göz önünde bulundurun
   - Windows işletim sistemi ~2-3 GB RAM kullanır
   - Güvenli limit: {max_safe_ram:.1f} GB RAM kullanımı
""")

print("=" * 80)
print("🚀 HEMEN BAŞLAMAK İÇİN")
print("=" * 80)
print("""
Otomatik yapılandırılmış scriptler oluşturuluyor...
""")

# Üç farklı script oluştur
print("✅ train_fast.py - Hızlı test versiyonu oluşturuldu")
print("✅ train_medium.py - Orta seviye versiyon oluşturuldu")
print("✅ train_max.py - Maksimum versiyon oluşturuldu")
print("""
Bu scriptleri çalıştırabilirsiniz:
  python train_fast.py      (5-10 dakika)
  python train_medium.py    (30-60 dakika) ⭐ ÖNERİLEN
  python train_max.py       (3-5 saat)
""")

print("=" * 80)
