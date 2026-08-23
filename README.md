# nanochat — Windows CPU adaptation

> [!IMPORTANT]
> Bu depo, [Andrej Karpathy'nin `nanochat` projesinin](https://github.com/karpathy/nanochat) resmi olmayan bir Windows/CPU adaptasyonudur. Orijinal proje ve temel mimari Andrej Karpathy'ye aittir. Bu depo upstream değildir ve upstream ile otomatik olarak senkronize edilmez.

Amaç, nanochat kodunu düşük bütçeli Windows bilgisayarlarda incelemek ve küçük Tiny Shakespeare deneyleri çalıştırmaktır. Bu çalışma üretime hazır bir sohbet modeli, upstream hız/kalite eşdeğeri veya ChatGPT düzeyi bir model iddiasında bulunmaz.

## Kaynak ve sahiplik

| Alan | Bilgi |
| --- | --- |
| Orijinal proje | [`karpathy/nanochat`](https://github.com/karpathy/nanochat) |
| Orijinal geliştirici | [Andrej Karpathy](https://github.com/karpathy) |
| Bu depo | Topluluk tarafından sürdürülen Windows/CPU adaptasyonu |
| Orijinal README kopyası | [`README_ORIGINAL.md`](README_ORIGINAL.md) |
| Lisans | Upstream MIT lisansı; [`LICENSE`](LICENSE) |

Bu adaptasyonun hangi upstream commit'inden ayrıldığı henüz makine tarafından okunabilir biçimde kaydedilmemiştir. Sonuçları karşılaştırırken yalnızca depo adı yerine commit SHA kullanın.

## Gerçekte ne değişti?

| Alan | Upstream yol | Bu adaptasyondaki değişiklik |
| --- | --- | --- |
| Model dtype | `nanochat/gpt.py` | `nanochat/gpt_cpu.py`, modeli ve rotary tablolarını FP32'ye çevirir |
| Optimizer | Muon / distributed AdamW | CPU wrapper'ı tüm parametreler için standart AdamW döndürür |
| Eğitim | Upstream GPU pipeline | Tiny Shakespeare üzerinde küçük, tek süreçli CPU scriptleri |
| Windows kullanımı | Birincil hedef değil | PowerShell odaklı başlangıç ve tanılama scriptleri |
| Kaynak ölçümü | Upstream raporlama | `benchmark_cpu.py` ile süre, token/s ve süreç RSS ölçümü |

“CPU optimizasyonu” burada kernel seviyesinde hızlandırma anlamına gelmez. Temel uyumluluk değişikliği BF16/mixed-dtype yolunu FP32'ye taşımak, dağıtık optimizer'ı kaldırmak ve modeli küçültmektir. FP32 daha geniş donanım uyumluluğu sağlar; fakat daha fazla bellek kullanabilir ve daha hızlı olacağı garanti değildir.

## Durum

- **Çalışan deney yolları:** küçük CPU eğitimi, checkpoint kaydı, örnek metin üretimi ve sistem tanılama.
- **Deneysel:** `train_fast.py`, `train_medium.py` ve `train_max.py` presetleri. Süreleri makineye göre değişir; dosyalardaki süre yorumları benchmark sonucu değildir.
- **Upstream yolu:** `speedrun.sh`, çoklu GPU değerlendirmeleri ve web sohbet akışı upstream kapsamındadır; Windows CPU desteğinin kanıtı olarak sunulmaz.
- **Doğrulanmamış:** upstream ile kalite/parite, “1000x” hız farkı ve sabit RAM/süre iddiaları. Karşılaştırılabilir kayıt üretilene kadar bunlar sonuç tablosuna yazılmamalıdır.

## Windows CPU kurulumu

Python 3.10+ ve Rust toolchain gerekir. Depodaki `pyproject.toml` / `uv.lock` upstream'in CUDA odaklı geliştirme çözümünü korur; Windows CPU için aşağıdaki ayrı kurulum yolu kullanılır.

```powershell
git clone https://github.com/cebrailbagatarhan/nanochat-windows-cpu.git
cd nanochat-windows-cpu

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-windows-cpu.txt
maturin develop --release
```

Kurulumu kısa bir eğitimle doğrulayın:

```powershell
python quick_test.py
```

Tiny Shakespeare eğitimi ve checkpoint'ten üretim:

```powershell
python train_cpu_shakespeare.py
python inference_cpu.py
```

İlk çalıştırmada Tiny Shakespeare verisi ve gerekli tokenizer dosyaları indirilebilir.

## Eğitim presetleri

Preset adları kalite seviyesi değil, kaynak tüketimi sıralamasıdır. Gerçek parametre sayısı tokenizer vocabulary boyutuna bağlıdır ve script tarafından çalışma anında yazdırılır.

| Script | Katman | Embedding | Adım | Amaç |
| --- | ---: | ---: | ---: | --- |
| `quick_test.py` | 2 | 128 | 10 | Kurulum kontrolü |
| `train_fast.py` | 2 | 128 | 50 | Kısa deney |
| `train_medium.py` | 6 | 384 | 200 | Daha büyük CPU deneyi |
| `train_max.py` | 8 | 512 | 500 | Uzun ve bellek yoğun deney |

Bu presetler aynı kaliteyi veya süreyi farklı makinelerde garanti etmez.

## Ölçüm

Karşılaştırılabilir sonuç üretmek için sentetik ve sabit tohumlu benchmark aracı eklendi:

```powershell
python benchmark_cpu.py --implementation upstream --mode train
python benchmark_cpu.py --implementation upstream-fp32 --mode train
python benchmark_cpu.py --implementation adaptation --mode train
```

Araç commit, donanım, PyTorch sürümü, parametre sayısı, token/s, süre, son loss ve peak RSS alanlarını JSON olarak yazar. Ölçüm protokolü ve doldurulacak sonuç tablosu için [`BENCHMARKS.md`](BENCHMARKS.md) dosyasına bakın. Bu depoda henüz aynı makinede alınmış, tekrarlanabilir upstream-vs-adaptation sonucu yayımlanmamıştır; tablo bu nedenle bilinçli olarak boş bırakılmıştır.

## Proje haritası

| Yol | Rol |
| --- | --- |
| `nanochat/gpt.py` | Depoda tutulan upstream model yolu |
| `nanochat/gpt_cpu.py` | FP32/CPU wrapper'ı |
| `train_cpu_shakespeare.py` | Küçük CPU eğitim akışı |
| `quick_test.py` | On adımlık smoke test |
| `inference_cpu.py` | Checkpoint ile etkileşimli üretim |
| `sistem_analizi.py` | CPU/RAM tahminleri; benchmark sonucu değildir |
| `benchmark_cpu.py` | Tekrarlanabilir sentetik benchmark |
| `README_ORIGINAL.md` | Upstream README'nin depoda tutulan kopyası |

## Sınırlamalar

- CPU eğitimi yavaştır; küçük deneyler temsilî sonuç üretir, genel amaçlı sohbet modeli üretmez.
- Tiny Shakespeare dar ve İngilizce bir corpus'tur.
- FP32, BF16'ya göre yaklaşık iki kat parametre/aktivasyon belleği gerektirebilir.
- Sentetik benchmark, gerçek veri hazırlama ve tokenizer maliyetini ölçmez.
- `upstream` benchmark'ının bir CPU'da dtype/operatör hatası vermesi mümkündür; bu da uyumluluk sonucudur ve başarısızlık olarak kaydedilmelidir.

## Katkı

Bir performans sonucu eklerken ham JSONL çıktısını, commit SHA'yı ve eksiksiz komutu birlikte gönderin. Sadece tahmini süre veya tek bir ekran görüntüsü karşılaştırılabilir benchmark kabul edilmez.

