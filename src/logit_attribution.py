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
    """Compute [layer, head] attribution for the first answer token after the key."""
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
        # The metric is the first value token's logit, i.e. the next-token output
        # immediately after the key. Keep all token scores above for diagnostics.
        scores.append(token_scores[0].detach().cpu())

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


def attribute_layers(model, tokenizer, trigger, answer):
    """Attribute each block's attention and MLP outputs to the first value-token logit."""
    base = require_gpt2(model)
    input_device = base.transformer.wte.weight.device
    trigger_ids = tokenizer(trigger, add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
    if not trigger_ids or not answer_ids:
        raise ValueError("Both trigger and answer must produce at least one token.")
    input_ids = torch.tensor([trigger_ids + answer_ids], device=input_device)
    target_ids = torch.tensor(answer_ids, device=input_device)
    position_start = len(trigger_ids) - 1
    position_end = len(trigger_ids) + len(answer_ids) - 1

    attention_outputs, mlp_outputs, final_residual, handles = {}, {}, {}, []

    def capture_output(destination, layer_index):
        def hook(module, inputs, output):
            # GPT2Attention returns a tuple; GPT2MLP returns its residual vector.
            destination[layer_index] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    for layer, block in enumerate(base.transformer.h):
        handles.append(block.attn.register_forward_hook(capture_output(attention_outputs, layer)))
        handles.append(block.mlp.register_forward_hook(capture_output(mlp_outputs, layer)))

    def capture_final_residual(module, inputs):
        final_residual["value"] = inputs[0].detach()
    handles.append(base.transformer.ln_f.register_forward_pre_hook(capture_final_residual))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    final_device = final_residual["value"].device
    final_positions = torch.arange(position_start, position_end, device=final_device)
    residual = final_residual["value"][0, final_positions]
    ln = base.transformer.ln_f
    scale = torch.rsqrt(residual.var(dim=-1, unbiased=False, keepdim=True) + ln.eps)
    direction = base.lm_head.weight[target_ids.to(base.lm_head.weight.device)].to(final_device)

    def component_score(component):
        positions = torch.arange(position_start, position_end, device=component.device)
        component = component[0, positions].to(final_device)
        component = (component - component.mean(dim=-1, keepdim=True)) * scale * ln.weight
        return torch.einsum("td,td->t", component, direction)[0].detach().cpu().item()

    attention_scores = [component_score(attention_outputs[layer]) for layer in range(base.config.n_layer)]
    mlp_scores = [component_score(mlp_outputs[layer]) for layer in range(base.config.n_layer)]
    return np.column_stack([attention_scores, mlp_scores])


def attribute_accumulated_residual(model, tokenizer, trigger, answer):
    """Track the first value-token logit through each pre-attention and mid-layer residual."""
    base = require_gpt2(model)
    input_device = base.transformer.wte.weight.device
    trigger_ids = tokenizer(trigger, add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
    if not trigger_ids or not answer_ids:
        raise ValueError("Both trigger and answer must produce at least one token.")
    input_ids = torch.tensor([trigger_ids + answer_ids], device=input_device)
    target_id = torch.tensor(answer_ids[0], device=input_device)
    target_position = len(trigger_ids) - 1

    resid_pre, attention_outputs, final_residual, handles = {}, {}, {}, []
    for layer, block in enumerate(base.transformer.h):
        def capture_pre(module, inputs, layer_index=layer):
            resid_pre[layer_index] = inputs[0].detach()

        def capture_attention(module, inputs, output, layer_index=layer):
            attention_outputs[layer_index] = (output[0] if isinstance(output, tuple) else output).detach()

        handles.append(block.register_forward_pre_hook(capture_pre))
        handles.append(block.attn.register_forward_hook(capture_attention))

    def capture_final_residual(module, inputs):
        final_residual["value"] = inputs[0].detach()
    handles.append(base.transformer.ln_f.register_forward_pre_hook(capture_final_residual))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    final_device = final_residual["value"].device
    final_position = torch.tensor(target_position, device=final_device)
    final_vector = final_residual["value"][0, final_position]
    ln = base.transformer.ln_f
    scale = torch.rsqrt(final_vector.var(unbiased=False) + ln.eps)
    direction = base.lm_head.weight[target_id.to(base.lm_head.weight.device)].to(final_device)

    def logit_attribution(residual):
        position = torch.tensor(target_position, device=residual.device)
        residual = residual[0, position].to(final_device)
        residual = (residual - residual.mean()) * scale * ln.weight
        return torch.dot(residual, direction).detach().cpu().item()

    labels, scores = [], []
    for layer in range(base.config.n_layer):
        pre = resid_pre[layer]
        # In a GPT-2 block this is exactly the residual stream after attention and
        # before the MLP: hidden_states = residual_pre + attention_output. Under
        # Accelerate device_map hooks, these two forward-hook captures can be on
        # adjacent GPUs, so explicitly combine them on the final-LN device.
        mid = pre.to(final_device) + attention_outputs[layer].to(final_device)
        labels.extend([f"L{layer} pred", f"L{layer} mid"])
        scores.extend([logit_attribution(pre), logit_attribution(mid)])
    return labels, np.asarray(scores)


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
        cbar_kws={"label": "Direct contribution to first correct value-token logit"},
    )
    plt.xlabel("Attention head")
    plt.ylabel("Transformer layer")
    plt.title("Direct Logit Attribution: Attention Heads → Memorised Value")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_layer_attribution(scores, path):
    """Plot direct logit attribution of attention and MLP outputs per layer."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    layers = np.arange(scores.shape[0])
    plt.figure(figsize=(max(10, scores.shape[0] * 0.8), 6))
    plt.axhline(0, color="black", linewidth=0.9)
    plt.plot(layers, scores[:, 0], marker="o", linewidth=2.2, label="Attention output")
    plt.plot(layers, scores[:, 1], marker="s", linewidth=2.2, label="MLP output")
    plt.xticks(layers, [f"L{layer}" for layer in layers])
    plt.xlabel("Transformer layer")
    plt.ylabel("Direct contribution to first correct value-token logit")
    plt.title("Layer Attribution: Attention and MLP Contributions (First Value Token)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_accumulated_residual_plot(labels, scores, path):
    """Save the IOI-style logit lens curve over accumulated residual states."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    positions = np.arange(len(scores))
    plt.figure(figsize=(max(12, len(scores) * 0.65), 6))
    plt.plot(positions, scores, marker="o", linewidth=2.2, color="#1f77b4")
    plt.axhline(0, color="black", linewidth=0.9)
    for boundary in range(2, len(scores), 2):
        plt.axvline(boundary - 0.5, color="#bbbbbb", linewidth=0.7, alpha=0.7)
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.xlabel("Accumulated residual-stream state")
    plt.ylabel("Direct contribution to first correct value-token logit")
    plt.title("Logit Attribution from Accumulated Residual Stream")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_head_list(head_list, n_layers, n_heads):
    """Parse a comma-separated ``layer.head`` list, e.g. ``9.6,9.9``."""
    selected = []
    for item in head_list.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layer, head = (int(value) for value in item.split(".", maxsplit=1))
        except ValueError as error:
            raise ValueError(f"Invalid head '{item}'. Use the format layer.head, e.g. 9.6.") from error
        if not (0 <= layer < n_layers and 0 <= head < n_heads):
            raise ValueError(f"Head {item} is outside this model's range: L0-{n_layers - 1}, H0-{n_heads - 1}.")
        selected.append((layer, head))
    if not selected:
        raise ValueError("Provide at least one head in --attention_heads.")
    return selected


def save_attention_patterns(model, tokenizer, trigger, answer, selected_heads, path):
    """Render full query × key attention maps for manually selected GPT-2 heads."""
    base = require_gpt2(model)
    input_device = base.transformer.wte.weight.device
    trigger_ids = tokenizer(trigger, add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
    input_ids = torch.tensor([trigger_ids + answer_ids], device=input_device)

    # Newer Transformers defaults may use SDPA, which does not expose attention
    # probabilities. Force eager attention only for this visualisation pass.
    original_implementation = getattr(model.config, "_attn_implementation", None)
    original_base_implementation = getattr(base.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"
    base.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        model.config._attn_implementation = original_implementation
        base.config._attn_implementation = original_base_implementation

    if outputs.attentions is None or any(pattern is None for pattern in outputs.attentions):
        raise RuntimeError("The installed Transformers version did not return attention probabilities.")

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(len(selected_heads), 1, figsize=(max(11, len(tokens) * 0.62), 4.8 * len(selected_heads)))
    axes = np.atleast_1d(axes)
    for axis, (layer, head) in zip(axes, selected_heads):
        pattern = outputs.attentions[layer][0, head].detach().float().cpu().numpy()
        image = axis.imshow(pattern, cmap="viridis", vmin=0, vmax=max(float(pattern.max()), 1e-8), aspect="auto")
        axis.set_title(f"Attention pattern — L{layer}.H{head}")
        axis.set_xlabel("Key position (source token)")
        axis.set_ylabel("Query position")
        axis.set_xticks(range(len(tokens)), tokens, rotation=65, ha="right", fontsize=8)
        axis.set_yticks(range(len(tokens)), tokens, fontsize=8)
        fig.colorbar(image, ax=axis, label="Attention probability")
    fig.suptitle("Selected attention heads on key + value sequence", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GPT-2 head-level direct logit attribution for memorised values")
    parser.add_argument("--model_path", required=True, help="Fine-tuned GPT-2 checkpoint path")
    parser.add_argument("--trigger", required=True, help="Key / recall prompt")
    parser.add_argument("--answer", required=True, help="Expected memorised value")
    parser.add_argument("--peft_path", default=None, help="Optional separate LoRA adapter path")
    parser.add_argument("--output_image", default="head_logit_attribution.png", help="Saved layer-by-head heatmap")
    parser.add_argument("--output_json", default=None, help="Optional detailed JSON output")
    parser.add_argument("--top_k", type=int, default=5, help="Heads printed from each end of the ranking")
    parser.add_argument("--attention_heads", default=None, help="Comma-separated heads to visualise, e.g. 9.6,9.9")
    parser.add_argument("--attention_image", default="attention_patterns.png", help="Saved attention-pattern figure")
    parser.add_argument("--layer_attribution_image", default="layer_logit_attribution.png", help="Saved attention-vs-MLP attribution plot")
    parser.add_argument("--accumulated_residual_image", default="accumulated_residual_logit_attribution.png", help="Saved accumulated-residual attribution plot")
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

    layer_scores = attribute_layers(model, tokenizer, args.trigger, args.answer)
    save_layer_attribution(layer_scores, args.layer_attribution_image)
    print(f"Layer attribution saved to: {args.layer_attribution_image}")

    residual_labels, residual_scores = attribute_accumulated_residual(model, tokenizer, args.trigger, args.answer)
    save_accumulated_residual_plot(residual_labels, residual_scores, args.accumulated_residual_image)
    print(f"Accumulated residual attribution saved to: {args.accumulated_residual_image}")

    if args.attention_heads:
        base = require_gpt2(model)
        selected_heads = parse_head_list(args.attention_heads, base.config.n_layer, base.config.n_head)
        save_attention_patterns(model, tokenizer, args.trigger, args.answer, selected_heads, args.attention_image)
        print(f"Attention patterns saved to: {args.attention_image}")

    if args.output_json:
        payload = {
            "trigger": args.trigger, "answer": args.answer,
            "answer_tokens": result["answer_tokens"], "answer_token_ids": result["answer_token_ids"],
            "target_logits": result["target_logits"],
            "attribution_target": "first_value_token",
            "head_attribution": scores.tolist(),
            "per_token_head_attribution": result["per_token_scores"].tolist(),
            "layer_attribution": {
                "columns": ["attention", "mlp"],
                "scores": layer_scores.tolist(),
            },
            "accumulated_residual_attribution": {
                "labels": residual_labels,
                "scores": residual_scores.tolist(),
            },
            "top_positive_heads": positive, "top_negative_heads": negative,
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        print(f"Detailed attribution saved to: {args.output_json}")


if __name__ == "__main__":
    main()
