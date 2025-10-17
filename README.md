# 🎭 nanochat - Windows CPU Edition# nanochat



> Andrej Karpathy'nin nanochat projesinin Windows CPU için optimize edilmiş versiyonu![nanochat logo](dev/nanochat.png)



[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)> The best ChatGPT that $100 can buy.

[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.



## ✨ Özellikler## Quick start



- ✅ **CPU Uyumlu:** BFloat16 → Float32 dönüşümü ile Windows CPU'da çalışırThe fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

- 🎯 **3 Seviyeli Eğitim:** Bilgisayar kapasitesine göre optimize edilmiş (Fast/Medium/Max)

- 📊 **Sistem Analizi:** Otomatik RAM ve CPU analizi ile öneriler```bash

- 🇹🇷 **Türkçe Dokümantasyon:** Detaylı kullanım kılavuzlarıbash speedrun.sh

- 🚀 **Test Edildi:** 12 core CPU, 15.7GB RAM sistemde başarıyla çalıştı```



## 🎯 Hızlı BaşlangıçAlternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):



```powershell```bash

# 1. Repoyu klonlascreen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

git clone https://github.com/cebrailbagatarhan/nanochat-windows-cpu.git```

cd nanochat-windows-cpu

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

# 2. Bağımlılıkları kur (uv gerekli)

pip install uv```bash

uv venvpython -m scripts.chat_web

uv pip install -r requirements.txt```



# 3. Rust tokenizer'ı derleAnd then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

cd rustbpe

cargo build --release---

cd ..

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

# 4. Sistem analizini çalıştır

python sistem_analizi.py---



# 5. Hızlı test (30 saniye)You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

python quick_test.py

---

# 6. Medium seviye eğitim (30-60 dakika, önerilen)

python train_medium.py- Characters: 333,989

- Lines: 8,304

# 7. Modeli test et- Files: 44

python test_model.py- Tokens (approx): 83,497

```- Dependencies (uv.lock lines): 2,004



## 📊 Eğitim Seviyeleri| Metric          | BASE     | MID      | SFT      | RL       |

|-----------------|----------|----------|----------|----------|

### ⚡ Fast (5-10 dakika)| CORE            | 0.2219   | -        | -        | -        |

- 2 katman, 128 boyut| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |

- ~17M parametre| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |

- ~300MB RAM| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |

- Hızlı test için ideal| HumanEval       | -        | 0.0671   | 0.0854   | -        |

| MMLU            | -        | 0.3111   | 0.3151   | -        |

### ⭐ Medium (30-60 dakika) - ÖNERİLEN| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

- 6 katman, 384 boyut

- ~61M parametreTotal wall clock time: 3h51m

- ~2-3GB RAM

- Dengeli sonuç---



### 🚀 Max (3-5 saat)(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

- 8 katman, 512 boyut

- ~92M parametre## Bigger models

- ~4-5GB RAM

- Maksimum kaliteUnsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.



## 📁 Proje YapısıThat said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:



``````bash

nanochat-windows-cpu/...

├── nanochat/# you'll need to download more data shards for pretraining

│   ├── gpt_cpu.py          # CPU uyumlu GPT wrapper# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,

│   ├── gpt.py              # Orijinal GPT modeli# divide by 250 million to get number of shards. todo need to improve this...

│   └── tokenizer.py        # Tokenizerpython -m nanochat.dataset -n 450 &

├── train_fast.py           # Hızlı eğitim...

├── train_medium.py         # Orta seviye (önerilen)# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:

├── train_max.py            # Maksimum eğitimtorchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16

├── test_model.py           # Model test scripti...

├── sistem_analizi.py       # Sistem analizi# make sure to use the same later during midtraining:

├── quick_test.py           # 30 saniyelik testtorchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16

├── README_CPU.md           # CPU dokümantasyonu```

├── WINDOWS_REHBER.md       # Windows kılavuzu

└── BASLARKEN.md            # Başlangıç rehberiThat's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensates by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

```

And a bit more about computing environments that will run nanochat:

## 🎨 Kullanım Örneği

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.

```python- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.

# Model eğit- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.

python train_medium.py- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.



# Test et## Questions

python test_model.py

nanochat is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

# Örnek çıktı:

# 📝 Prompt: 'ROMEO:'```bash

# ROMEO:files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt

# What light through yonder window breaks?```

# It is the east, and Juliet is the sun.

```This includes all py, rs, html, toml, sh files, excludes the `rustbpe/target` folder, and chooses the cxml output format. Everything is written to the `packaged.txt` file, which atm measures ~330KB (i.e. well below ~100K tokens for a state of the art LLM), and ~8K lines of code in 45 files.



## 🔧 Teknik DetaylarAlternatively, I recommend using [DeepWiki](https://deepwiki.com/) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.



### Orijinal nanochat## Tests

- 8x H100 GPU

- BFloat16 precisionI haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

- Muon optimizer

- 560M parametre```bash

- $100 / 4 saatpython -m pytest tests/test_rustbpe.py -v -s

```

### Windows CPU Edition

- CPU only (GPU yok)## Contributing

- Float32 precision

- AdamW optimizernanochat is nowhere finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

- 17M-92M parametre

- Ücretsiz / 10dk-5saat## Acknowledgements



## 📚 Dokümantasyon- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.

- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.

- [README_CPU.md](README_CPU.md) - CPU uyarlama detayları- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.

- [WINDOWS_REHBER.md](WINDOWS_REHBER.md) - Windows kurulum kılavuzu- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.

- [BASLARKEN.md](BASLARKEN.md) - Hızlı başlangıç rehberi- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.



## 🙏 Teşekkürler## Cite



Bu proje [Andrej Karpathy'nin nanochat](https://github.com/karpathy/nanochat) projesinin Windows CPU uyarlamasıdır.If you find nanochat helpful in your research cite simply as:



## 📜 Lisans```bibtex

@misc{nanochat,

MIT License - Orijinal proje ve bu uyarlama  author = {Andrej Karpathy},

  title = {nanochat: The best ChatGPT that $100 can buy},

## 🤝 Katkıda Bulunma  year = {2025},

  publisher = {GitHub},

Pull request'ler memnuniyetle karşılanır! Büyük değişiklikler için önce issue açın.  url = {https://github.com/karpathy/nanochat}

}

## 📊 Test Sonuçları```



**Sistem:** 12 CPU çekirdek, 15.7 GB RAM## License



- ✅ **Quick test:** 30 saniye, loss 11.09 → 3.32MIT

- ✅ **Fast:** 10 dakika, 17M param
- ✅ **Medium:** 1.2 dakika, 61M param, loss 5.79 → 2.37
- 🔄 **Max:** Test ediliyor, 92M param

## 📞 İletişim

GitHub: [@cebrailbagatarhan](https://github.com/cebrailbagatarhan)

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!
