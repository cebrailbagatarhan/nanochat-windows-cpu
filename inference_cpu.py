"""
CPU'da eğitilmiş nanochat modelini güvenli checkpoint yükleme ile kullan.
"""

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from nanochat.gpt_cpu import CPUCompatibleGPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


CHECKPOINT_CANDIDATES = (
    "model_max.pt",
    "model_medium.pt",
    "model_fast.pt",
    "best_model.pt",
)


def resolve_checkpoint(explicit_path: str | None, data_dir: Path) -> Path:
    """Açık yolu veya mevcut en güçlü eğitim checkpoint'ini seç."""
    if explicit_path:
        checkpoint_path = Path(explicit_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
        return checkpoint_path

    for filename in CHECKPOINT_CANDIDATES:
        checkpoint_path = data_dir / filename
        if checkpoint_path.is_file():
            return checkpoint_path

    searched = ", ".join(str(data_dir / name) for name in CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        "Eğitilmiş checkpoint bulunamadı. Aranan yollar: " + searched
    )


def load_model(checkpoint_path: Path):
    """Yalnızca tensor ve temel Python tipleri içeren checkpoint'i yükle."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint güvenli biçimde yüklenemedi. Eski checkpoint config "
            "nesnesi içeriyor olabilir; güncel eğitim scriptiyle yeniden üretin."
        ) from exc

    required_keys = {"config", "model_state_dict", "step", "val_loss"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "Checkpoint alanları eksik: " + ", ".join(sorted(missing_keys))
        )

    config_data = checkpoint["config"]
    if not isinstance(config_data, dict):
        raise TypeError("Checkpoint 'config' alanı dictionary olmalıdır")

    config = GPTConfig(**config_data)
    model = CPUCompatibleGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def generate_text(
    model,
    config,
    tokenizer,
    prompt,
    max_tokens=200,
    temperature=0.8,
    top_k=40,
):
    """Prompt'tan autoregressive metin üret."""
    if temperature <= 0:
        raise ValueError("temperature pozitif olmalıdır")

    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_tokens):
            context_crop = (
                context
                if context.size(1) <= config.sequence_len
                else context[:, -config.sequence_len:]
            )
            logits, _ = model(context_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1)),
                )
                logits[logits < values[:, [-1]]] = -float("inf")

            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            context = torch.cat([context, next_token], dim=1)

    return tokenizer.decode(context[0].tolist())


def parse_args():
    parser = argparse.ArgumentParser(description="nanochat CPU inference")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint yolu; verilmezse mevcut en güçlü checkpoint seçilir",
    )
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path.home() / ".cache" / "nanochat" / "shakespeare"

    print("=" * 80)
    print("🎭 nanochat CPU Inference")
    print("=" * 80)

    try:
        checkpoint_path = resolve_checkpoint(args.checkpoint, data_dir)
        print(f"\n📂 Model yükleniyor: {checkpoint_path}")
        model, config, checkpoint = load_model(checkpoint_path)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"\n❌ {exc}")
        print("   Önce bir eğitim scriptini çalıştırın.")
        raise SystemExit(1) from exc

    tokenizer = get_tokenizer()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise RuntimeError(
            "Tokenizer/checkpoint vocabulary uyumsuzluğu: "
            f"{tokenizer.get_vocab_size()} != {config.vocab_size}"
        )

    print("   ✅ Model yüklendi")
    print(f"   📊 Val loss: {checkpoint['val_loss']:.4f}")
    print(f"   📈 Eğitim step: {checkpoint['step']}")

    examples = [
        "ROMEO:",
        "JULIET:",
        "To be or not to be,",
        "Friends, Romans, countrymen,",
        "What light through yonder window",
    ]

    print("\n🎮 İNTERAKTİF MOD")
    print("  Çıkış: q / quit / exit")
    print("  Örnekler: örnek")

    while True:
        prompt = input("\n💬 Prompt: ").strip()
        if prompt.lower() in {"q", "quit", "exit"}:
            print("👋 Görüşürüz!")
            break
        if prompt.lower() in {"örnek", "example", "examples"}:
            for index, example in enumerate(examples, 1):
                print(f"   {index}. {example}")
            continue
        if not prompt:
            print("⚠️ Boş prompt")
            continue

        try:
            generated = generate_text(
                model=model,
                config=config,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print("\n" + "-" * 80)
            print(generated)
            print("-" * 80)
        except Exception as exc:
            print(f"❌ Üretim hatası: {exc}")


if __name__ == "__main__":
    main()
