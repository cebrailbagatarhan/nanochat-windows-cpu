"""
Windows için Hızlı Başlangıç Menüsü
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║            🎭 nanochat - Windows CPU Edition (12 core, 15.7GB)            ║
╚════════════════════════════════════════════════════════════════════════════╝

Windows bilgisayarınızda CPU ile çalışan özelleştirilmiş nanochat!
✨ BİLGİSAYARINIZA ÖZEL OPTİMİZE EDİLDİ ✨

📚 MEVCUT KOMUTLAR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  DEMO & ANALİZ
   python sistem_analizi.py            # Sistem kapasitesi analizi
   python demo_windows.py              # Tokenizer demo
   python durum_raporu.py              # Sistem durumu
   python quick_test.py                # Hızlı test (30 saniye)

2️⃣  EĞİTİM (Bilgisayarınıza Göre Optimize!)
   python train_fast.py                # ⚡ 5-10 dakika (2 layer, 128 dim)
   python train_medium.py              # ⭐ 30-60 dakika (6 layer, 384 dim) ÖNERİLEN
   python train_max.py                 # 🚀 3-5 saat (8 layer, 512 dim) MAKSİMUM
   python train_cpu_shakespeare.py     # Orta seviye (varsayılan)

3️⃣  INFERENCE (Metin Üretme)
   python inference_cpu.py             # Eğitilmiş modelle sohbet
   
   🎮 İnteraktif mod - promptlar yazıp metin üretin!

4️⃣  REHBERLER
   code WINDOWS_REHBER.md              # Detaylı kullanım kılavuzu
   code BASLARKEN.md                   # Hızlı başlangıç

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 BİLGİSAYARINIZA GÖRE OPTİMİZE EDİLMİŞ SEVYELER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CPU Uyumlu GPT modeli (nanochat/gpt_cpu.py)
   - BFloat16 → Float32 dönüşümü
   - CPU için optimize edilmiş forward pass
   - Basitleştirilmiş optimizer (AdamW)

✅ ÜÇ SEVİYE EĞİTİM (Bilgisayarınıza özel):
   
   ⚡ FAST: 2 layer, 128 dim, ~17M param, ~300MB RAM
      → 5-10 dakika, hızlı test
   
   ⭐ MEDIUM: 6 layer, 384 dim, ~61M param, ~2-3GB RAM
      → 30-60 dakika, dengeli sonuç (ÖNERİLEN!)
   
   🚀 MAX: 8 layer, 512 dim, ~92M param, ~4-5GB RAM
      → 3-5 saat, en iyi sonuç (bilgisayarınızın limiti)

✅ İnteraktif Inference
   - Eğitilmiş modeli yükle
   - İstediğiniz prompt ile metin üret
   - Temperature ve top-k ayarları

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 HIZLI BAŞLANGIÇ (Bilgisayarınıza Özel):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Sistem analizi yap (öneriler için):
   python sistem_analizi.py

2. Hızlı test (30 saniye):
   python quick_test.py

3. Dengeli eğitim (30-60 dakika) - ÖNERİLEN:
   python train_medium.py

4. Metin üret:
   python inference_cpu.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  PERFORMANS NOTLARI (12 Core CPU, 15.7GB RAM):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Eğitim süreleri (bilgisayarınıza göre):
  - Quick test: ~30 saniye ✅
  - Fast (50 steps): ~5-10 dakika ✅
  - Medium (200 steps): ~30-60 dakika ⭐ ÖNERİLEN
  - Max (500 steps): ~3-5 saat 🚀 MAKSİMUM
  
Model kalitesi:
  - Fast: Çok basit (sadece test)
  - Medium: Demo seviyesi (gösterilebilir)
  - Max: İyi (bilgisayarınızın limiti)
  - Gerçek ChatGPT: 8xH100 GPU gerekir

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 İPUÇLARI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. İlk kez kullanıyorsanız: python sistem_analizi.py
2. Hızlı denemek için: python train_fast.py (5-10 dakika)
3. Dengeli sonuç için: python train_medium.py (30-60 dakika) ⭐
4. Maksimum kalite için: python train_max.py (3-5 saat)
5. Eğitim sırasında diğer programları kapatın (RAM için)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌟 ÖZELLİK: Bilgisayarınıza özel optimize edildi!
   12 CPU çekirdeği, 15.7 GB RAM için en uygun ayarlar.

╚════════════════════════════════════════════════════════════════════════════╝
""")

import sys
if len(sys.argv) > 1:
    cmd = sys.argv[1].lower()
    if cmd == 'demo':
        import subprocess
        subprocess.run([sys.executable, 'demo_windows.py'])
    elif cmd == 'train':
        import subprocess
        subprocess.run([sys.executable, 'train_medium.py'])  # Varsayılan orta seviye
    elif cmd == 'fast':
        import subprocess
        subprocess.run([sys.executable, 'train_fast.py'])
    elif cmd == 'medium':
        import subprocess
        subprocess.run([sys.executable, 'train_medium.py'])
    elif cmd == 'max':
        import subprocess
        subprocess.run([sys.executable, 'train_max.py'])
    elif cmd == 'inference':
        import subprocess
        subprocess.run([sys.executable, 'inference_cpu.py'])
    elif cmd == 'status':
        import subprocess
        subprocess.run([sys.executable, 'durum_raporu.py'])
    elif cmd == 'analyze':
        import subprocess
        subprocess.run([sys.executable, 'sistem_analizi.py'])
else:
    print("\n💡 Kullanım: python start.py [analyze|demo|fast|medium|max|inference|status]")
    print("   veya komutları manuel çalıştırın ↑")
