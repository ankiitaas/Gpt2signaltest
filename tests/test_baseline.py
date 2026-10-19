import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from baseline import ablate, explain, select_positions, target_metrics


def test_ranked_and_seeded_controls():
    scores = [0.1, 0.9, 0.4, 0.9]
    assert select_positions(scores, 2, "top") == [1, 3]
    assert select_positions(scores, 2, "bottom") == [0, 2]
    selected = select_positions(scores, 2, "random", 42)
    assert selected == select_positions(scores, 2, "random", 42)
    assert len(set(selected)) == 2
    with pytest.raises(ValueError):
        select_positions(scores, 5, "top")


def test_ablation_preserves_unselected_tokens_and_original():
    embeddings = torch.ones(1, 4, 3)
    changed = ablate(embeddings, [1, 3])
    assert changed.shape == embeddings.shape
    assert changed[:, [1, 3]].count_nonzero() == 0
    assert torch.equal(changed[:, [0, 2]], embeddings[:, [0, 2]])
    assert embeddings.sum() == 12
    assert torch.equal(ablate(embeddings, []), embeddings)


def test_target_metrics_keep_requested_target():
    metrics = target_metrics(torch.tensor([0.0, 2.0]), 0)
    assert metrics["argmax_id"] == 1
    assert metrics["logit"] == 0
    assert metrics["probability"] == pytest.approx(0.1192029)


def test_tiny_decoder_gradients_attention_and_noop():
    torch.manual_seed(7)
    config = GPT2Config(vocab_size=32, n_positions=16, n_embd=16, n_layer=2, n_head=2,
                        attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config).eval().requires_grad_(False)
    input_ids = torch.tensor([[2, 4, 6, 8]])
    target, clean, embeddings, scores, _ = explain(model, input_ids, target=3)
    assert target == 3
    for values in scores.values():
        assert len(values) == 4
        assert torch.isfinite(torch.tensor(values)).all()
        assert sum(values) > 0
    assert sum(scores["attention"]) == pytest.approx(1.0)
    with torch.no_grad():
        noop = model(inputs_embeds=ablate(embeddings, []), use_cache=False).logits[0, -1]
    assert target_metrics(noop, target)["log_probability"] == pytest.approx(clean["log_probability"])
    repeat = explain(model, input_ids, target=3)
    assert repeat[3] == scores
