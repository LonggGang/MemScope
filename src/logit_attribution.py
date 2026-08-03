"""Direct logit attribution of GPT-2 attention heads for key-value recall."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, peft_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device_map = "auto" if torch.cuda.is_available() else None
    adapter_path = peft_path
    base_model_path = model_path
    if adapter_path is None and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        adapter_path = model_path
        with open(os.path.join(model_path, "adapter_config.json"), encoding="utf-8") as file:
            base_model_path = json.load(file)["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map=device_map
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def require_gpt2(model):
    """Return the GPT-2 model under a PEFT wrapper, if present."""
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    if not hasattr(base, "transformer") or not hasattr(base.transformer, "h"):
        raise NotImplementedError(
            "Head attribution currently supports GPT-2 checkpoints "
            "(transformer.h.<layer>.attn.c_proj)."
        )
    return base


def split_head_results(merged_head_values, c_proj, n_heads):
    """Project GPT-2's concatenated attention values separately for each head."""
    batch, seq, d_model = merged_head_values.shape
    if d_model % n_heads:
        raise ValueError("GPT-2 attention width is not divisible by its number of heads.")
    d_head = d_model // n_heads
    values = merged_head_values.reshape(batch, seq, n_heads, d_head)
    # GPT-2 Conv1D uses a [input, output] weight matrix.
    weight = c_proj.weight.reshape(n_heads, d_head, d_model)
    return torch.einsum("bshd,hdo->bsho", values, weight)


def attribute_heads(model, tokenizer, trigger, answer):
    """Compute mean direct attribution [layer, head] over all answer tokens."""
    base = require_gpt2(model)
    # ``device_map=auto`` may shard GPT-2 blocks across multiple GPUs. Inputs must
    # start on the embedding device, while every later index is created on the
    # device of the tensor being indexed (rather than assuming a single GPU).
    input_device = base.transformer.wte.weight.device

    # Match SFT formatting: ``<trigger> <answer>``.  Teacher forcing lets us score
    # every token in a multi-token value, not only its first generated token.
    trigger_ids = tokenizer(trigger, add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
    if not trigger_ids or not answer_ids:
        raise ValueError("Both trigger and answer must produce at least one token.")
    input_ids = torch.tensor([trigger_ids + answer_ids], device=input_device)
    target_ids = torch.tensor(answer_ids, device=input_device)
    position_start = len(trigger_ids) - 1
    position_end = len(trigger_ids) + len(answer_ids) - 1

    head_inputs, final_residual, handles = {}, {}, []
    for layer, block in enumerate(base.transformer.h):
        def capture_head_input(module, inputs, layer_index=layer):
            head_inputs[layer_index] = inputs[0].detach()
        handles.append(block.attn.c_proj.register_forward_pre_hook(capture_head_input))

    def capture_final_residual(module, inputs):
        final_residual["value"] = inputs[0].detach()
    handles.append(base.transformer.ln_f.register_forward_pre_hook(capture_final_residual))

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    # DLA holds the final LayerNorm scale fixed at the real residual stream. This
    # makes the residual-to-logit map additive, while preserving its actual scale.
    final_device = final_residual["value"].device
    final_positions = torch.arange(position_start, position_end, device=final_device)
    residual = final_residual["value"][0, final_positions]
    ln = base.transformer.ln_f
    scale = torch.rsqrt(residual.var(dim=-1, unbiased=False, keepdim=True) + ln.eps)
    direction = base.lm_head.weight[target_ids.to(base.lm_head.weight.device)].to(final_device)

    scores, per_token_scores = [], []
    for layer, block in enumerate(base.transformer.h):
        results = split_head_results(head_inputs[layer], block.attn.c_proj, base.config.n_head)
        layer_positions = torch.arange(position_start, position_end, device=results.device)
        # Move only this small [answer_token, head, d_model] attribution tensor to
        # the final-LN device. This supports Accelerate's multi-GPU device maps.
        results = results[0, layer_positions].to(final_device)
        results = (results - results.mean(dim=-1, keepdim=True)) * scale[:, None, :] * ln.weight
        token_scores = torch.einsum("thd,td->th", results, direction)
        per_token_scores.append(token_scores.detach().cpu())
        scores.append(token_scores.mean(dim=0).detach().cpu())

    logit_positions = torch.arange(position_start, position_end, device=outputs.logits.device)
    target_logits = outputs.logits[0, logit_positions].gather(
        1, target_ids.to(outputs.logits.device)[:, None]
    ).squeeze(1)
    return {
        "scores": torch.stack(scores).numpy(),
        "per_token_scores": torch.stack(per_token_scores).numpy(),
        "answer_token_ids": answer_ids,
        "answer_tokens": tokenizer.convert_ids_to_tokens(answer_ids),
        "target_logits": target_logits.cpu().tolist(),
    }


def top_heads(scores, k, largest=True):
    indices = np.argsort(scores.ravel())
    if largest:
        indices = indices[::-1]
    return [
        {"layer": int(i // scores.shape[1]), "head": int(i % scores.shape[1]), "score": float(scores.ravel()[i])}
        for i in indices[:k]
    ]


def save_heatmap(scores, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    limit = max(float(np.abs(scores).max()), 1e-8)
    plt.figure(figsize=(max(9, scores.shape[1] * 0.72), max(6, scores.shape[0] * 0.48)))
    sns.heatmap(
        scores, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit,
        xticklabels=[f"H{i}" for i in range(scores.shape[1])],
        yticklabels=[f"L{i}" for i in range(scores.shape[0])],
        cbar_kws={"label": "Mean direct contribution to correct value-token logit"},
    )
    plt.xlabel("Attention head")
    plt.ylabel("Transformer layer")
    plt.title("Direct Logit Attribution: Attention Heads → Memorised Value")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="GPT-2 head-level direct logit attribution for memorised values")
    parser.add_argument("--model_path", required=True, help="Fine-tuned GPT-2 checkpoint path")
    parser.add_argument("--trigger", required=True, help="Key / recall prompt")
    parser.add_argument("--answer", required=True, help="Expected memorised value")
    parser.add_argument("--peft_path", default=None, help="Optional separate LoRA adapter path")
    parser.add_argument("--output_image", default="head_logit_attribution.png", help="Saved layer-by-head heatmap")
    parser.add_argument("--output_json", default=None, help="Optional detailed JSON output")
    parser.add_argument("--top_k", type=int, default=5, help="Heads printed from each end of the ranking")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.peft_path)
    result = attribute_heads(model, tokenizer, args.trigger, args.answer)
    scores = result["scores"]
    save_heatmap(scores, args.output_image)
    positive, negative = top_heads(scores, args.top_k), top_heads(scores, args.top_k, largest=False)
    print("\nTop positive heads (increase correct value-token logits):")
    for item in positive:
        print(f"  L{item['layer']}.H{item['head']}: {item['score']:+.4f}")
    print("Top negative heads (decrease correct value-token logits):")
    for item in negative:
        print(f"  L{item['layer']}.H{item['head']}: {item['score']:+.4f}")
    print(f"\nHeatmap saved to: {args.output_image}")

    if args.output_json:
        payload = {
            "trigger": args.trigger, "answer": args.answer,
            "answer_tokens": result["answer_tokens"], "answer_token_ids": result["answer_token_ids"],
            "target_logits": result["target_logits"],
            "mean_head_attribution": scores.tolist(),
            "per_token_head_attribution": result["per_token_scores"].tolist(),
            "top_positive_heads": positive, "top_negative_heads": negative,
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        print(f"Detailed attribution saved to: {args.output_json}")


if __name__ == "__main__":
    main()
