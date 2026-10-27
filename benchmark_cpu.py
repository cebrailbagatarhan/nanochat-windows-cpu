"""Reproducible CPU compatibility and throughput benchmark for this adaptation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.gpt_cpu import CPUCompatibleGPT


MIB = 1024 * 1024


class PeakRssSampler:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    config = GPTConfig(
        sequence_len=args.sequence_length,
        vocab_size=args.vocab_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_kv_head=args.kv_heads,
        n_embd=args.embedding_size,
    )
    model: torch.nn.Module
    if args.implementation == "adaptation":
        model = CPUCompatibleGPT(config)
    else:
        model = GPT(config)
        if args.implementation == "upstream-fp32":
            model = model.to(torch.float32)
    return model.to("cpu")


def train_benchmark(
    model: torch.nn.Module, args: argparse.Namespace
) -> tuple[dict[str, float], float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randint(
        args.vocab_size,
        (args.batch_size, args.sequence_length),
        generator=generator,
    )
    targets = torch.randint(
        args.vocab_size,
        (args.batch_size, args.sequence_length),
        generator=generator,
    )

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(args.warmup_steps):
        step()

    final_loss = float("nan")
    started = time.perf_counter()
    for _ in range(args.steps):
        final_loss = step()
    elapsed = time.perf_counter() - started
    processed_tokens = args.batch_size * args.sequence_length * args.steps
    return {
        "elapsed_seconds": elapsed,
        "processed_tokens": processed_tokens,
        "train_tokens_per_second": processed_tokens / elapsed,
    }, final_loss


@torch.inference_mode()
def inference_benchmark(
    model: torch.nn.Module, args: argparse.Namespace
) -> tuple[dict[str, float], None]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    initial_length = min(32, args.sequence_length)
    context = torch.randint(
        args.vocab_size,
        (args.batch_size, initial_length),
        generator=generator,
    )

    def step(current: torch.Tensor) -> torch.Tensor:
        cropped = current[:, -args.sequence_length :]
        logits, _ = model(cropped)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return torch.cat((current, next_token), dim=1)

    for _ in range(args.warmup_steps):
        context = step(context)

    started = time.perf_counter()
    for _ in range(args.steps):
        context = step(context)
    elapsed = time.perf_counter() - started
    generated_tokens = args.batch_size * args.steps
    return {
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed,
    }, None


def base_result(args: argparse.Namespace) -> dict[str, object]:
    memory = psutil.virtual_memory()
    return {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_sha(),
        "implementation": args.implementation,
        "mode": args.mode,
        "seed": args.seed,
        "threads": args.threads,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "system_ram_gib": memory.total / (1024**3),
        "config": {
            "layers": args.layers,
            "embedding_size": args.embedding_size,
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "vocab_size": args.vocab_size,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
        },
    }


def emit(result: dict[str, object], output: Path | None) -> None:
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    print(encoded)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        choices=("upstream", "upstream-fp32", "adaptation"),
        required=True,
    )
    parser.add_argument("--mode", choices=("train", "inference"), default="train")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.embedding_size % args.heads:
        parser.error("--embedding-size must be divisible by --heads")
    if args.heads % args.kv_heads:
        parser.error("--heads must be divisible by --kv-heads")
    for name in ("layers", "embedding_size", "heads", "kv_heads", "vocab_size", "sequence_length", "batch_size", "steps", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    result = base_result(args)

    try:
        with PeakRssSampler() as memory:
            model = build_model(args)
            result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
            runner = train_benchmark if args.mode == "train" else inference_benchmark
            metrics, final_loss = runner(model, args)
        result.update(metrics)
        result["peak_rss_mib"] = memory.peak_bytes / MIB
        result["final_loss"] = final_loss
        result["status"] = "passed"
        emit(result, args.output)
        return 0
    except Exception as error:  # The upstream CPU failure is itself a benchmark result.
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        emit(result, args.output)
        return 1


if __name__ == "__main__":
    sys.exit(main())

