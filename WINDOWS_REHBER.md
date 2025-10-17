# nanochat - Windows Kullanım Rehberi

## ✅ Kurulum Tamamlandı!

Tebrikler! nanochat başarıyla kuruldu. Ancak bu proje **GPU gerektirir** ve özellikle **8xH100** gibi güçlü GPU'lar için tasarlanmıştır.

## 🖥️ Şu Anda Ne Yapabilirsiniz?

### 1. Kodu İncelemek ve Öğrenmek ✨

```powershell
# Sanal ortamı aktifleştir
.\.venv\Scripts\Activate.ps1

# Kodu incele
code .
```

**İncelenecek önemli dosyalar:**
- `nanochat/gpt.py` - Transformer modeli
- `scripts/base_train.py` - Eğitim scripti
- `nanochat/tokenizer.py` - Tokenizer
- `nanochat/engine.py` - Inference engine

### 2. Tokenizer'ı Test Etmek 🔤

```powershell
# Sanal ortamı aktifleştir
.\.venv\Scripts\Activate.ps1

# Demo scripti çalıştır
python demo_windows.py
```

Tokenizer zaten çalışıyor! Metinleri token'lara dönüştürebilir.

### 3. Bulut GPU Kullanmak (Önerilen) ☁️

**nanochat'i gerçekten kullanmak için:**

#### Seçenek A: Lambda Cloud (En Kolay)
1. [Lambda Cloud](https://lambda.ai/service/gpu-cloud)'a kaydol
2. 8xH100 instance kirala (~$24/saat)
3. Linux sunucuda:
   ```bash
   git clone https://github.com/karpathy/nanochat.git
   cd nanochat
   bash speedrun.sh
   ```
4. 4 saat bekle
5. $100'a kendi ChatGPT'ne sahip ol!

#### Seçenek B: Google Colab (Ücretsiz/Düşük Maliyet)
- Ücretsiz GPU ile küçük deneyler yapabilirsiniz
- Colab Pro+ ile daha güçlü GPU'lar

#### Seçenek C: Diğer Cloud Sağlayıcılar
- AWS (Amazon EC2 P5 instances)
- Azure (NC-series)
- Google Cloud Platform (A100/H100 VMs)

### 4. Eğitilmiş Model İndirmek 📥

Başkaları nanochat eğitmiş olabilir:

1. Hugging Face'i kontrol et:
   - https://huggingface.co/models?search=nanochat

2. İndirdikten sonra inference yapabilirsiniz:
   ```powershell
   python -m scripts.chat_cli
   ```

## 🚫 CPU'da Neden Çalışmıyor?

nanochat şu nedenlerle CPU'da çalışamaz:

1. **BFloat16 Gereksinimi**: Model bfloat16 kullanıyor, CPU'lar bunu desteklemez
2. **Hız**: GPU'da 4 saat süren şey CPU'da aylar sürer
3. **Bellek**: Büyük modeller CPU RAM'ine sığmaz
4. **Tasarım**: Kod özellikle GPU için optimize edilmiş

## 📊 Maliyet Karşılaştırması

| Platform | GPU | Süre | Maliyet |
|----------|-----|------|---------|
| Lambda Cloud | 8xH100 | 4 saat | ~$100 |
| AWS EC2 | 8xH100 | 4 saat | ~$120 |
| Google Cloud | 8xH100 | 4 saat | ~$110 |
| Colab Pro+ | 1xA100 | Çok uzun | ~$50/ay |
| Kendi CPU | - | İmkansız | - |

## 🎓 Öğrenme Kaynakları

1. **Andrej Karpathy'nin Videoları**:
   - [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
   - [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A)

2. **nanochat Discussion**:
   - https://github.com/karpathy/nanochat/discussions/1

3. **Kodu Okumak**:
   - nanochat çok temiz ve iyi dokümante edilmiş
   - Her satırı okuyup anlayabilirsiniz

## 💡 Pratik Öneriler

### Bütçeniz Yoksa:
1. Kodu okuyun ve öğrenin
2. Küçük deneyler için Colab kullanın
3. Eğitilmiş modelleri indirin ve test edin

### Bütçeniz Varsa ($100-200):
1. Lambda Cloud'dan GPU kiralayın
2. speedrun.sh'yi çalıştırın
3. Kendi modelinizle oynayın
4. Farklı hiperparametreleri deneyin

### Çok İleri Gitmek İsterseniz:
1. Daha büyük modeller eğitin (depth=26 veya depth=30)
2. Farklı datasetler deneyin
3. Katkıda bulunun (Pull Request)

## 🤝 Topluluk

- **Discord**: https://discord.gg/3zy8kqD9Cp (#nanochat kanalı)
- **GitHub Discussions**: https://github.com/karpathy/nanochat/discussions
- **GitHub Issues**: https://github.com/karpathy/nanochat/issues

## 📝 Özet

✅ Windows'ta kurulum yapıldı
✅ Kod incelenebilir
✅ Tokenizer çalışıyor
❌ Model eğitimi CPU'da mümkün değil
✅ Bulut GPU ile mükemmel çalışır!

**Tavsiye**: $100 bütçeniz varsa Lambda Cloud'dan GPU kiralayın ve gerçek deneyimi yaşayın!
