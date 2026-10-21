#decoderonlyllm signal test

baseline: which internal signals explain a decoder-only model's next-token prediction, and at what extraction cost?

## Setup and run

Use Python 3.11–3.13. CPU execution is the reproducibility default. The first run downloads GPT-2 Small (124M parameters); allow approximately 3 GB disk space for the environment and weights.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python baseline.py
python baseline.py --prompt "The capital of France is" --target " Paris" --output outputs/france
```

Open this folder in VS Code, install the recommended Python extensions, select `.venv/bin/python`, and run the **GPT-2 baseline** debug configuration. Tests use a tiny randomly initialized GPT-2 and require no model download. They test the pipeline, not pretrained model quality.

## Protocol

- Three default smoke-test prompts; no training or generation sampling. Each prompt predicts one next token. The target defaults to the clean argmax and stays fixed across interventions. An explicit target must encode to exactly one token.
- Pinned Hugging Face model/tokenizer revision, pinned direct dependencies, seed 42, CPU float32, one thread, evaluation mode, deterministic PyTorch algorithms. `requirements-lock.txt` records the tested environment when available. Numerical identity across platforms is not guaranteed.
- Attention importance is the mean over all layers and heads of the final query's attention to each input token. This is an unsigned, target-independent heuristic, not attention rollout.
- Gradients differentiate the fixed target's **log probability** with respect to input token embeddings. Scores are gradient L2 norm and absolute signed gradient-times-input sum across embedding dimensions. Magnitude discards direction.
- For each signal, ablate the top and bottom `ceil(0.2 * token_count)` positions, plus 10 seeded random subsets of identical size. All positions, including the final query token, are eligible. Ties use token position. Random subsets are shared across methods for paired comparison.
- Ablation zeroes token embeddings before position embeddings are added. Sequence length, positions, and attention connectivity remain unchanged. This is not token deletion or a native mask token; GPT-2 has no native mask token. Zero embeddings can be out of distribution and selected positions can still carry contextual information in later layers.
- Save clean and perturbed target logits, natural-log probabilities, probabilities, prediction changes, token positions, scores, and timings. Drops equal clean minus perturbed; positive means the intervention reduced support for the target. Random trials are averaged within each prompt before aggregation across prompts.

## Outputs and interpretation

`outputs/baseline/metrics.json` contains all measurements and environment metadata; `summary.csv` contains mean log-probability drops and sample standard deviations across prompts. Timings separate model loading, forward/attention extraction, backward gradient extraction, and interventions; they are single-run wall-clock observations, not rigorous benchmarks. Gradient timing excludes its required forward pass. Random-control trial details are in JSON.

Large top-ranked drops relative to bottom/random controls are preliminary evidence under this intervention, not proof of causal explanation or general faithfulness. Three prompts are a smoke test, not a benchmark. Next steps: larger held-out prompt sets, multiple intervention baselines, signed attributions, activation patching, bootstrap uncertainty, and cost/faithfulness trade-offs. No attention tensors or weights are committed; results under `results/` are curated reproducibility artifacts.

## References

- [GPT-2 Small model card](https://huggingface.co/openai-community/gpt2)
- [Transformers GPT-2 documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)

Repository intended to remain private. Model weights retain their upstream license; no project redistribution license is assigned yet.
