# CPU benchmark protocol

Bu belge `nanochat/gpt.py` ile `nanochat/gpt_cpu.py` yollarını aynı makinede karşılaştırmak için kullanılır. Amaç önceden belirlenmiş bir sonuca ulaşmak değil; uyumluluk, throughput ve bellek maliyetini tekrarlanabilir biçimde kaydetmektir.

## Varyantlar

| Ad | Tanım |
| --- | --- |
| `upstream` | Depodaki `nanochat.gpt.GPT` sınıfı, dtype'a müdahale edilmeden |
| `upstream-fp32` | Aynı sınıf, yalnızca `model.to(torch.float32)` uygulanmış uyumluluk baseline'ı; unmodified upstream değildir |
| `adaptation` | `nanochat.gpt_cpu.CPUCompatibleGPT` |

Training benchmark'ında üç varyant için de aynı standart AdamW kullanılır. Böylece ölçüm model forward/backward yoluna odaklanır; adaptasyonun optimizer tercihi ayrıca README'de açıklanır.

## Sabit protokol

1. Aynı commit ve temiz çalışma ağacını kullanın.
2. Güç modunu ve arka plan uygulamalarını sabit tutun.
3. Aynı Python, PyTorch ve thread sayısını kullanın.
4. Her varyantı üç kez çalıştırın; ilk run'ı otomatik olarak dışlamayın.
5. Komutları değiştirmeden JSONL çıktısını commit edin.
6. Başarısız upstream çalışmasını silmeyin; `status=failed` uyumluluk bulgusudur.

Önerilen PowerShell komutları:

```powershell
New-Item -ItemType Directory -Force benchmark-results | Out-Null

1..3 | ForEach-Object {
  python benchmark_cpu.py --implementation upstream --mode train --seed 42 --threads 8 --output benchmark-results/train.jsonl
  python benchmark_cpu.py --implementation upstream-fp32 --mode train --seed 42 --threads 8 --output benchmark-results/train.jsonl
  python benchmark_cpu.py --implementation adaptation --mode train --seed 42 --threads 8 --output benchmark-results/train.jsonl
}

1..3 | ForEach-Object {
  python benchmark_cpu.py --implementation upstream --mode inference --seed 42 --threads 8 --output benchmark-results/inference.jsonl
  python benchmark_cpu.py --implementation upstream-fp32 --mode inference --seed 42 --threads 8 --output benchmark-results/inference.jsonl
  python benchmark_cpu.py --implementation adaptation --mode inference --seed 42 --threads 8 --output benchmark-results/inference.jsonl
}
```

Varsayılan model küçük tutulur: 4 katman, 256 embedding, 4 head, 4.096 vocabulary, 128 sequence length. Farklı bir konfigürasyon kullanırsanız bütün argümanları sonuçla birlikte raporlayın.

## Sonuç tablosu

Henüz aynı makine ve aynı protokolle üretilmiş karşılaştırılabilir sonuç commit edilmedi.

| Commit | Donanım | Mod | Varyant | Durum | Parametre | Token/s | Peak RSS MiB | Final loss |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| — | — | train | upstream | not measured | — | — | — | — |
| — | — | train | upstream-fp32 | not measured | — | — | — | — |
| — | — | train | adaptation | not measured | — | — | — | — |
| — | — | inference | upstream | not measured | — | — | — | — |
| — | — | inference | upstream-fp32 | not measured | — | — | — | — |
| — | — | inference | adaptation | not measured | — | — | — | — |

## Yorumlama

- `train_tokens_per_second`, batch × sequence × ölçülen step sayısı üzerinden hesaplanır.
- `generated_tokens_per_second`, üretilen batch tokenı üzerinden hesaplanır; KV cache kullanılmaz.
- `peak_rss_mib`, yalnızca bu Python sürecinin örneklenen resident belleğidir. Sistem toplam RAM'i değildir.
- Sentetik tokenlar veri yükleme/tokenizer maliyetini kapsamaz.
- Tek run farkı performans kanıtı değildir; medyan ve yayılımı birlikte raporlayın.

