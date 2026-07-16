"""J-space readout (logit lens) and steering-vector construction/injection.

This module holds the whole interpretability mechanism -- reading a layer's activation as
words, building a steering direction two different ways, and turning that direction into a
forward hook. Both steering methods return a plain [d_model] tensor so everything
downstream (build_steering_vector's caller, make_steering_hook) is agnostic to which one
produced it.
"""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from config import Config


def get_resid_post_at_layer(model: HookedTransformer, prompt: str, layer: int) -> torch.Tensor:
    """Residual stream after layer `layer`, at the prompt's last token position.

    `cache["resid_post", layer]` comes out as [batch=1, seq_pos, d_model] (one vector per
    input token); we index down to [1, d_model] at the *last* position because that's the
    position whose activation is about to become "the next token's logits" -- i.e. the
    position the J-space readout (and CLAUDE.md's `resid[:, -1]`) is defined on.

    names_filter restricts run_with_cache to caching only this one hook point instead of
    every activation in the model -- on CPU, for Qwen2.5-1.5B, caching everything would be
    needlessly slow and memory-heavy for a script that only ever reads one layer.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt, names_filter=lambda name: name == hook_name)
    resid = cache["resid_post", layer]  # [1, seq_pos, d_model]
    return resid[:, -1, :]  # [1, d_model]


def logit_lens_readout(model: HookedTransformer, resid_bd: torch.Tensor, top_k: int) -> list[tuple[str, float]]:
    """Decode a [1, d_model] residual-stream activation into its top-k vocabulary tokens.

    This is the logit lens: apply the model's own final layernorm, then its unembedding
    matrix -- exactly the same path the real final-layer activation takes on its way to
    becoming next-token logits. Skipping ln_final would read a raw, un-normalized
    direction and give a distorted top-k that doesn't match what the model would actually
    say if this were its output layer.
    """
    with torch.no_grad():
        logits = model.unembed(model.ln_final(resid_bd))  # [1, d_vocab]
    top_vals, top_idx = logits.squeeze(0).topk(top_k)  # [top_k], [top_k]
    return [(model.to_string(int(idx)), float(val)) for idx, val in zip(top_idx, top_vals)]


def build_actadd_vector(model: HookedTransformer, layer: int, pos_prompt: str, neg_prompt: str) -> torch.Tensor:
    """ActAdd steering vector: difference of resid_post at `layer` between a pos/neg prompt pair.

    Both prompts are read at the same layer and the same last-token convention as the
    readout, so the difference isolates "what this layer represents differently between
    these two prompts" as a single [d_model] direction, independent of either prompt's
    absolute activation scale.
    """
    pos = get_resid_post_at_layer(model, pos_prompt, layer).squeeze(0)  # [d_model]
    neg = get_resid_post_at_layer(model, neg_prompt, layer).squeeze(0)  # [d_model]
    return pos - neg  # [d_model]


def _single_token_id(model: HookedTransformer, value: str, flag_name: str) -> int:
    """model.to_single_token raises a bare AssertionError with no mention of which flag or
    model caused it; re-raise as a clear, actionable error instead of letting an unusable
    --pos/--neg value silently fall back to a default or surface as a confusing traceback.
    """
    try:
        return model.to_single_token(value)
    except AssertionError as e:
        raise ValueError(
            f"--{flag_name} {value!r} is not a single token for {model.cfg.model_name}'s "
            f"tokenizer, but --steering-method token_diff requires a single-token concept "
            f"for both --pos and --neg. Pick a single-token string, or use "
            f"--steering-method actadd for multi-word concepts."
        ) from e


def build_token_diff_vector(model: HookedTransformer, pos_token: str, neg_token: str) -> torch.Tensor:
    """"Point at a token" steering vector: difference of two unembedding columns.

    model.W_U has shape [d_model, d_vocab]; column i is the residual-stream direction that
    most directly raises token i's logit. This method needs no forward pass at all -- it's
    a pure lookup into the model's weights. _single_token_id raises if the string isn't
    exactly one BPE token, which is the correctness check we want here (a silent multi-token
    split would quietly build a vector for the wrong token).
    """
    pos_id = _single_token_id(model, pos_token, "pos")
    neg_id = _single_token_id(model, neg_token, "neg")
    return model.W_U[:, pos_id] - model.W_U[:, neg_id]  # [d_model]


def build_steering_vector(model: HookedTransformer, cfg: Config, layer: int) -> torch.Tensor:
    """Single dispatch point for the two steering methods -- see module docstring."""
    if cfg.steering_method == "actadd":
        return build_actadd_vector(model, layer, cfg.pos, cfg.neg)
    if cfg.steering_method == "token_diff":
        return build_token_diff_vector(model, cfg.pos, cfg.neg)
    raise ValueError(f"Unknown steering method: {cfg.steering_method!r}")


def make_steering_hook(steering_vector: torch.Tensor, coefficient: float):
    """Build a TransformerLens forward hook that adds `coefficient * steering_vector`.

    Hook signature is (activation, hook) -> replacement_activation. `resid_post` here has
    shape [batch, seq_pos, d_model]: seq_pos is the full prompt length on the first
    (prefill) forward pass, then 1 on every subsequent incremental-decoding step once the
    KV cache takes over. The hook fires on both, which is exactly why wrapping
    model.generate in `with model.hooks(fwd_hooks=[...]):` steers every generated token,
    not just the prompt's own forward pass. steering_vector ([d_model]) broadcasts against
    resid_post ([batch, seq_pos, d_model]) via ordinary tensor broadcasting -- no reshape
    needed.
    """
    def hook_fn(resid_post: torch.Tensor, hook) -> torch.Tensor:
        return resid_post + coefficient * steering_vector

    return hook_fn
