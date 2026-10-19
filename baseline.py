"""Fixed-target next-token attribution and token-embedding ablation."""

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".cache" / "huggingface"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PROMPTS = ["The capital of France is", "The opposite of hot is", "The scientist published the results in a"]


def select_positions(scores, count, strategy, seed=0):
    if not 0 <= count <= len(scores):
        raise ValueError("Ablation count must be between zero and token count")
    if strategy == "random":
        return sorted(random.Random(seed).sample(range(len(scores)), count))
    if strategy not in {"top", "bottom"}:
        raise ValueError("Unknown selection strategy")
    order = sorted(range(len(scores)), key=lambda index: (
        -scores[index] if strategy == "top" else scores[index], index
    ))
    return sorted(order[:count])


def ablate(embeddings, positions):
    changed = embeddings.detach().clone()
    changed[:, positions, :] = 0
    return changed


def target_metrics(logits, target):
    log_probability = logits.log_softmax(dim=-1)[target].item()
    return {"logit": logits[target].item(), "log_probability": log_probability,
            "probability": math.exp(log_probability), "argmax_id": logits.argmax().item()}


def explain(model, input_ids, target=None):
    started = time.perf_counter()
    embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    result = model(inputs_embeds=embeddings, output_attentions=True, use_cache=False)
    logits = result.logits[0, -1]
    target = logits.argmax().item() if target is None else target
    clean = target_metrics(logits, target)
    attention = torch.stack([layer[0, :, -1, :] for layer in result.attentions]).mean((0, 1))
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    gradient, = torch.autograd.grad(logits.log_softmax(-1)[target], embeddings)
    scores = {
        "attention": attention.detach().tolist(),
        "gradient_l2": gradient[0].norm(dim=-1).tolist(),
        "gradient_x_input_abs": (gradient * embeddings).sum(-1)[0].abs().detach().tolist(),
    }
    return target, clean, embeddings.detach(), scores, {
        "forward_attention_seconds": forward_seconds,
        "gradient_seconds": time.perf_counter() - started,
    }


def evaluate(model, tokenizer, prompt, fraction, trials, seed, target_text=None):
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    if not 1 <= input_ids.shape[1] <= model.config.n_positions:
        raise ValueError("Prompt must contain 1 to model context-limit tokens")
    target = None
    if target_text is not None:
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if len(target_ids) != 1:
            raise ValueError("--target must encode to exactly one token; include leading space if needed")
        target = target_ids[0]
    target, clean, embeddings, scores, timings = explain(model, input_ids, target)
    count = max(1, math.ceil(input_ids.shape[1] * fraction))
    rows = []
    started = time.perf_counter()
    for method, importance in scores.items():
        for strategy in ("top", "bottom", "random"):
            for trial in range(trials if strategy == "random" else 1):
                positions = select_positions(importance, count, strategy, seed + trial)
                with torch.no_grad():
                    logits = model(inputs_embeds=ablate(embeddings, positions), use_cache=False).logits[0, -1]
                changed = target_metrics(logits, target)
                rows.append({"method": method, "strategy": strategy, "trial": trial,
                             "positions": positions, "count": count, **changed,
                             "logit_drop": clean["logit"] - changed["logit"],
                             "log_probability_drop": clean["log_probability"] - changed["log_probability"],
                             "probability_drop": clean["probability"] - changed["probability"],
                             "prediction_changed": clean["argmax_id"] != changed["argmax_id"]})
    timings["ablation_seconds"] = time.perf_counter() - started
    return {"prompt": prompt, "input_ids": input_ids[0].tolist(),
            "tokens": [tokenizer.decode([token]) for token in input_ids[0].tolist()],
            "target_id": target, "target_text": tokenizer.decode([target]),
            "clean": clean, "scores": scores, "ablations": rows, "timings": timings}


def summarize(examples):
    summary = []
    for method in examples[0]["scores"]:
        for strategy in ("top", "bottom", "random"):
            per_prompt = [statistics.mean(row["log_probability_drop"] for row in example["ablations"]
                         if row["method"] == method and row["strategy"] == strategy) for example in examples]
            summary.append({"method": method, "strategy": strategy,
                            "mean_log_probability_drop": statistics.mean(per_prompt),
                            "std_across_prompts": statistics.stdev(per_prompt) if len(per_prompt) > 1 else 0,
                            "prompt_count": len(per_prompt)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", help="Repeat for multiple prompts")
    parser.add_argument("--target", help="Single next-token text; default is clean argmax")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or args.random_trials < 1:
        parser.error("fraction must be in (0, 1], random-trials must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, attn_implementation="eager", use_safetensors=True
    ).float().cpu().eval()
    model.requires_grad_(False)
    load_seconds = time.perf_counter() - started
    examples = [evaluate(model, tokenizer, prompt, args.fraction, args.random_trials,
                         args.seed, args.target) for prompt in (args.prompt or PROMPTS)]
    report = {"schema_version": 1, "model": MODEL, "revision": REVISION,
              "seed": args.seed, "device": "cpu", "dtype": "float32",
              "threads": 1, "fraction": args.fraction, "random_trials": args.random_trials,
              "intervention": "zero token embeddings; preserve positions and attention mask",
              "gradient_objective": "fixed target log probability", "load_seconds": load_seconds,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")}},
              "examples": examples, "summary": summarize(examples)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report["summary"][0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["summary"])
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
