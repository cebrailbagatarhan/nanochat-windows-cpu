# 🎭 nanochat - Windows CPU Edition

> **Orijinal nanochat'in Windows bilgisayarlarda CPU ile çalışabilen özelleştirilmiş versiyonu**

## ✅ Yapılan Özelleştirmeler

### 🔧 Teknik Değişiklikler

1. **CPU Uyumlu GPT Modeli** (`nanochat/gpt_cpu.py`)
   - BFloat16 → Float32 dönüşümü (CPU bfloat16 desteklemez)
   - CPU için optimize edilmiş forward pass
   - BFloat16 assertion'ları kaldırıldı
   - Basitleştirilmiş optimizer (Muon yerine AdamW)

2. **Shakespeare Eğitim Scripti** (`train_cpu_shakespeare.py`)
   - Otomatik veri indirme (Tiny Shakespeare dataset)
   - Küçük model yapılandırması (CPU için optimize)
   - Checkpoint kaydetme
   - Validation ve training loss takibi
   - En iyi modeli otomatik kaydetme

3. **İnteraktif Inference** (`inference_cpu.py`)
   - Eğitilmiş modeli yükleme
   - İnteraktif metin üretme
   - Temperature ve top-k sampling
   - Örnek promptlar

## 🚀 Hızlı Başlangıç

### 1. Sanal Ortamı Aktifleştir
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Sistem Testini Çalıştır (30 saniye)
```powershell
python quick_test.py
```

### 3. Model Eğit (15-30 dakika)
```powershell
python train_cpu_shakespeare.py
```

### 4. Metin Üret
```powershell
python inference_cpu.py
# veya belirli bir checkpoint:
python inference_cpu.py --checkpoint ~/.cache/nanochat/shakespeare/model_medium.pt
```

## 📚 Tüm Komutlar

| Komut | Açıklama | Süre |
|-------|----------|------|
| `python start.py` | Ana menü | - |
| `python quick_test.py` | Hızlı sistem testi | 30 saniye |
| `python demo_windows.py` | Tokenizer demo | 5 saniye |
| `python durum_raporu.py` | Sistem durumu | 5 saniye |
| `python train_cpu_shakespeare.py` | Shakespeare eğitimi | 15-30 dakika |
| `python inference_cpu.py` | Metin üretme | İnteraktif |

## ⚙️ Model Yapılandırması

### Default Ayarlar (train_cpu_shakespeare.py)
```python
config = GPTConfig(
    sequence_len=128,    # Context uzunluğu
    n_layer=6,           # Transformer katmanları
    n_head=6,            # Attention heads
    n_embd=384,          # Embedding boyutu
)

max_steps = 200         # Eğitim adımları
batch_size = 4          # Batch size
```

### Önerilen Ayarlamalar

**Hızlı Test:**
```python
max_steps = 50
n_layer = 2
n_embd = 128
```

**Denge:**
```python
max_steps = 500
n_layer = 4
n_embd = 256
```

**Daha İyi Sonuç (Yavaş):**
```python
max_steps = 1000
n_layer = 8
n_embd = 512
```

## 📊 Performans Beklentileri

### CPU Performansı (8 çekirdek)
- **Quick Test (10 steps)**: ~30 saniye
- **Default (100 steps, 4 layers)**: ~15-30 dakika
- **Extended (500 steps, 4 layers)**: ~2-3 saat
- **Large (1000 steps, 8 layers)**: ~6-8 saat

### GPU Karşılaştırması
- GPU eğitimi CPU eğitiminden belirgin biçimde hızlıdır; süre donanım, model ve batch ayarına göre ölçülmelidir.
- Orijinal speedrun ayrı veri, model ve dağıtık eğitim ayarları kullanır; bu CPU Shakespeare deneyiyle kalite açısından eşdeğer değildir.

## 💡 İpuçları

### Daha İyi Sonuçlar İçin

1. **max_steps artırın**: 100 → 500 → 1000
2. **Model büyütün**: n_layer=4 → 8, n_embd=256 → 512
3. **Daha fazla veri**: Shakespeare'den başka metinler ekleyin
4. **Learning rate ayarlayın**: Script içinde optimizer ayarları

### Eğitim Sırasında

- Loss azalıyorsa ✅ model öğreniyor
- Loss sabitlendiyse → max_steps artırın veya learning rate değiştirin
- Val loss artıyorsa → overfitting, regularization ekleyin

### Inference Ayarları

```python
generate_text(
    prompt="ROMEO:",
    max_tokens=200,      # Üretilecek token sayısı
    temperature=0.8,     # Yaratıcılık (0.5-1.5 arası)
    top_k=40            # Çeşitlilik kontrolü
)
```

- **temperature = 0.5**: Daha tutarlı, tekrarlayan
- **temperature = 1.0**: Dengeli
- **temperature = 1.5**: Daha yaratıcı, kaotik

## 📁 Dosya Yapısı

```
nanochat/
├── nanochat/
│   ├── gpt.py              # Orijinal GPT (GPU için)
│   ├── gpt_cpu.py          # ✨ CPU uyumlu GPT
│   └── tokenizer.py        # BPE tokenizer
├── scripts/
│   └── base_train.py       # Orijinal eğitim (GPU)
├── start.py                # ✨ Ana menü
├── quick_test.py           # ✨ Hızlı test
├── train_cpu_shakespeare.py # ✨ CPU eğitim scripti
├── inference_cpu.py        # ✨ Metin üretme
├── demo_windows.py         # ✨ Tokenizer demo
└── durum_raporu.py         # ✨ Durum raporu
```

✨ = Windows CPU için özel oluşturulmuş

## 🎯 Örnek Kullanım

### Eğitim
```powershell
# Sanal ortamı aktifleştir
.\.venv\Scripts\Activate.ps1

# Eğitimi başlat
python train_cpu_shakespeare.py

# Çıktı:
# 🎭 nanochat CPU Eğitimi
# 📥 Veri indiriliyor... ✅
# 🏗️ Model oluşturuluyor... ✅ yaklaşık 61M parametre
# 🚀 EĞİTİM BAŞLIYOR
# Step 0/100 | Loss: 11.0904 | 0.25s/step
# Step 20/100 | Loss: 7.3542 | 0.22s/step
# ...
```

### Inference
```powershell
python inference_cpu.py

# İnteraktif mod:
💬 Prompt: ROMEO:
🎯 Üretilen metin:
ROMEO: O, she doth teach the torches to burn bright!
...
```

## 🔧 Troubleshooting

### Problem: "Module not found"
```powershell
# Sanal ortamı aktifleştirdiğinizden emin olun
.\.venv\Scripts\Activate.ps1
```

### Problem: "Out of memory"
```python
# train_cpu_shakespeare.py içinde:
batch_size = 4 → 2  # Daha küçük batch
n_layer = 4 → 2     # Daha küçük model
```

### Problem: "Too slow"
```python
# Daha hızlı test için:
max_steps = 100 → 50
n_layer = 4 → 2
```

## 🌟 Özellikler

### ✅ Çalışıyor
- Tokenizer (BPE)
- CPU'da model eğitimi
- Checkpoint kaydetme
- Metin üretme
- İnteraktif inference
- Loss tracking

### ⚠️ Sınırlamalar
- CPU eğitimi yavaştır; hız farkı model ve donanıma göre ölçülmelidir
- Küçük modeller (bellek sınırı)
- BFloat16 yerine Float32 (hassasiyet kaybı minimal)

### ❌ Desteklenmiyor
- Multi-GPU eğitim
- Distributed training
- Muon optimizer (AdamW kullanılıyor)
- CUDA acceleration

## 📖 Daha Fazla Bilgi

- `WINDOWS_REHBER.md` - Detaylı Windows kullanım kılavuzu
- `BASLARKEN.md` - Hızlı başlangıç rehberi
- [Orijinal nanochat](https://github.com/karpathy/nanochat)
- [GitHub Discussion](https://github.com/karpathy/nanochat/discussions/1)

## 🎓 Öğrenme Kaynakları

1. **Andrej Karpathy Videoları**:
   - [Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
   - [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

2. **Kod İnceleyin**:
   - `nanochat/gpt_cpu.py` - Model yapısı
   - `train_cpu_shakespeare.py` - Eğitim loop
   - Kod çok temiz ve iyi dokümante!

## 🤝 Katkıda Bulunma

Bu CPU versiyonu resmi değil, topluluk adaptasyonudur. 
Orijinal projeye katkıda bulunmak için: https://github.com/karpathy/nanochat

## ⚖️ Lisans

Orijinal nanochat lisansına tabidir.

---

**🎭 Windows CPU'da nanochat deneyimi!**

Bu proje eğitim ve çalışma zamanı deneyi içindir; kısa Shakespeare koşuları genel amaçlı sohbet modeli kalitesi üretmez.

## CPU eğitim günlüğü özeti (çevrimdışı)

Mevcut `train_fast.py`, `train_medium.py`, `train_max.py` veya
`train_cpu_shakespeare.py` çıktısını bir metin dosyasına kaydedip eğitim
başlatmadan özetleyebilirsiniz:

```bash
python train_fast.py > train.log 2>&1
python -m scripts.cpu_log_summary train.log --json
python -m unittest discover -s tests -p 'test_cpu_log_summary.py' -v
```

Araç yalnızca mevcut metin dosyasını okur; model/checkpoint yüklemez, ağ
isteği yapmaz ve logu değiştirmez. Adım sayısı yalnızca loga yazılan adımları,
ortalama süre yalnızca loglanan adımların sürelerini ifade eder. Kısmi
loglardan eğitim tamamlandı sonucu çıkarılmaz.
