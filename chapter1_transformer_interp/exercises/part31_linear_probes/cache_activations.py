"""
Cache the model activations needed for [1.3.1] Linear Probes, so the probing exercises can be done
on a machine without a large GPU.

Run on a GPU node from this directory, e.g.

    python cache_activations.py --out ~/scratch/arena/part31_acts --stages got deception8b highstakes
    python cache_activations.py --out ~/scratch/arena/part31_acts --stages deception70b

Stages and outputs (one or more .pt files per stage in --out):

    got           Llama-2-13b-hf. got.pt: last-token activations at ALL layers for cities, sp_en_trans,
                  larger_than, neg_cities, plus reference few-shot / causal-intervention results.
    deception8b   Llama-3.1-8B-Instruct. deception8b.pt: instructed-pairs activations (mean over
                  assistant tokens, all layers, all 512 true facts in CSV order), per-token AI Liar
                  activations (all layers), plus reference steering generations.
    highstakes    Llama-3.1-8B-Instruct. highstakes_{train,test,ood_<cfg>}.pt: full-sequence layer-16
                  activations (padded to 256 tokens) for the models-under-pressure dataset.
    deception70b  Llama-3.3-70B-Instruct (int8). deception70b.pt: per-token layer-22 AI Liar
                  activations, plus reference scores from the deception-detection repo's detector.

Activation tensors are stored in the model's dtype (bf16, or fp16 for the int8 70B model); call
.float() after loading. "All layers" tensors have a layer axis where index l is the output of decoder
layer l, i.e. hidden_states[l + 1], matching the solutions' convention. Data ordering and the
train/test splits follow solutions.py exactly, so the same seeds reproduce the same splits locally.
"""

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

section_dir = Path(__file__).resolve().parent
exercises_dir = section_dir.parent
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part31_linear_probes.utils as utils  # noqa: E402

GOT_DATASETS = exercises_dir / "geometry-of-truth" / "datasets"
DD_ROOT = exercises_dir / "deception-detection"
DD_DATA = DD_ROOT / "data"

GOT_DATASET_NAMES = ["cities", "sp_en_trans", "larger_than", "neg_cities"]
PROBE_LAYER, INTERVENE_LAYER = 14, 8  # Llama-2-13b layers used in the solutions
HS_LAYER, HS_MAX_LEN, HS_N_TRAIN, HS_N_TEST = 16, 256, 1500, 500
HS_OOD_CONFIGS = ["anthropic_hh_balanced", "toolace_balanced"]

FEW_SHOT_PROMPT = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(obj, path: Path) -> None:
    t.save(obj, path)
    log(f"saved {path} ({path.stat().st_size / 1e9:.2f} GB)")


def load_model(name: str, **kwargs):
    log(f"loading {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    kwargs.setdefault("dtype", t.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", **kwargs)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log(f"loaded: {model.config.num_hidden_layers} layers, d_model={model.config.hidden_size}")
    return model, tokenizer


def cleanup() -> None:
    gc.collect()
    t.cuda.empty_cache()


def mm_direction(acts: t.Tensor, labels: t.Tensor) -> t.Tensor:
    """Difference-of-means direction, as MMProbe.from_data computes it."""
    return acts[labels == 1].mean(0) - acts[labels == 0].mean(0)


# ---------------------------------------------------------------------------------------------
# Stage: got (Llama-2-13b)
# ---------------------------------------------------------------------------------------------


@t.no_grad()
def last_token_all_layers(statements, model, tokenizer, batch_size=25) -> t.Tensor:
    """Same as solutions' extract_activations, but for all layers. Returns [n, n_layers, d_model]."""
    out = []
    for i in tqdm(range(0, len(statements), batch_size), desc="last-token acts"):
        batch = statements[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        hs = model(**inputs, output_hidden_states=True).hidden_states[1:]
        last = inputs["attention_mask"].sum(dim=1) - 1
        idx = t.arange(len(batch), device=last.device)
        out.append(t.stack([h[idx, last.to(h.device)].cpu() for h in hs], dim=1))
    return t.cat(out)


@t.no_grad()
def intervention_p_diffs(statements, model, tokenizer, direction, layers, scale, batch_size=32) -> t.Tensor:
    """P(TRUE) - P(FALSE) with scale * direction added at the last statement tokens (solutions' method)."""
    true_id = tokenizer.encode(" TRUE")[-1]
    false_id = tokenizer.encode(" FALSE")[-1]
    len_suffix = len(tokenizer.encode(" This statement is:"))
    p_diffs = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        queries = [FEW_SHOT_PROMPT + s + " This statement is:" for s in batch]
        inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        attn = inputs["attention_mask"]

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            seq_lens = attn.sum(dim=1)
            for b in range(hidden.shape[0]):
                end = seq_lens[b].item()
                for offset in [-len_suffix, -len_suffix - 1]:
                    pos = int(end + offset)
                    if 0 <= pos < hidden.shape[1]:
                        hidden[b, pos, :] += scale * direction.to(hidden.device)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        hooks = [model.model.layers[l].register_forward_hook(hook_fn) for l in layers] if scale != 0 else []
        try:
            logits = model(**inputs).logits
        finally:
            for h in hooks:
                h.remove()
        last = attn.sum(dim=1) - 1
        probs = logits[t.arange(len(batch), device=logits.device), last.to(logits.device)].softmax(-1)
        p_diffs.append((probs[:, true_id] - probs[:, false_id]).float().cpu())
    return t.cat(p_diffs)


def stage_got(args) -> None:
    model, tokenizer = load_model(args.got_model)
    result = {"model": args.got_model, "probe_layer": PROBE_LAYER, "intervene_layer": INTERVENE_LAYER, "datasets": {}}

    for name in GOT_DATASET_NAMES:
        df = pd.read_csv(GOT_DATASETS / f"{name}.csv")
        if args.limit:
            df = df.iloc[: args.limit]
        statements = df["statement"].tolist()
        log(f"got/{name}: {len(statements)} statements")
        result["datasets"][name] = {
            "statements": statements,
            "labels": t.tensor(df["label"].values, dtype=t.float32),
            "acts": last_token_all_layers(statements, model, tokenizer),
        }

    # Reference results for the few-shot / causal intervention exercises (these need the live model)
    ds = result["datasets"]
    sp_statements, sp_labels = ds["sp_en_trans"]["statements"], ds["sp_en_trans"]["labels"]
    keep = [s not in FEW_SHOT_PROMPT for s in sp_statements]
    sp_eval = [s for s, k in zip(sp_statements, keep) if k]
    sp_eval_labels = sp_labels[t.tensor(keep)]
    combined_acts = t.cat([ds["cities"]["acts"][:, PROBE_LAYER], ds["neg_cities"]["acts"][:, PROBE_LAYER]]).float()
    combined_labels = t.cat([ds["cities"]["labels"], ds["neg_cities"]["labels"]])
    direction = mm_direction(combined_acts, combined_labels)
    direction_hat = direction / direction.norm()
    scaled_direction = (direction @ direction_hat).item() * direction_hat
    layers = list(range(INTERVENE_LAYER, PROBE_LAYER + 1))

    reference = {"sp_eval_statements": sp_eval, "sp_eval_labels": sp_eval_labels, "intervention_layers": layers}
    for intervention, scale in [("none", 0.0), ("add", 1.0), ("subtract", -1.0)]:
        log(f"got/intervention: {intervention}")
        reference[f"p_diff_{intervention}"] = intervention_p_diffs(sp_eval, model, tokenizer, scaled_direction, layers, scale)
    reference["scaled_direction"] = scaled_direction
    result["reference"] = reference
    save(result, args.out / "got.pt")
    del model, result
    cleanup()


# ---------------------------------------------------------------------------------------------
# Stages: deception8b, highstakes (Llama-3.1-8B-Instruct)
# ---------------------------------------------------------------------------------------------


@t.no_grad()
def instructed_pair_acts(facts, model, tokenizer) -> tuple[t.Tensor, t.Tensor]:
    """Solutions' construct_instructed_pairs, for all layers. Returns honest, dishonest [n, n_layers, d]."""
    out = {"honest": [], "dishonest": []}
    prompts = {"honest": "You are an honest AI assistant.", "dishonest": "You are a deceptive AI assistant."}
    for fact in tqdm(facts, desc="instructed pairs"):
        words = fact.split(" ")
        fact_start = " ".join(words[:-5]) if len(words) > 5 else fact
        for key, sys_prompt in prompts.items():
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Please tell me a fact."},
                {"role": "assistant", "content": fact_start},
            ]
            _, tokens, attn_mask, det_mask = utils.build_detection_mask(messages, tokenizer)
            hs = model(tokens.to(model.device), output_hidden_states=True).hidden_states[1:]
            if det_mask.sum() > 0:
                acts = [h[0][det_mask.to(h.device)].mean(0) for h in hs]
            else:
                acts = [h[0][attn_mask[0].sum() - 1] for h in hs]
            out[key].append(t.stack(acts).cpu())
    return t.stack(out["honest"]), t.stack(out["dishonest"])


def ai_liar_dialogues(limit: int | None = None):
    with open(DD_DATA / "how_to_catch_an_ai_liar" / "goal_directed_lying.json") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    dialogues = []
    for i, item in enumerate(data):
        for variant, key in [("honest", "normal_instruction"), ("deceptive", "deceive_instruction")]:
            messages = [
                {"role": "system", "content": item[key]},
                {"role": "user", "content": item["question_prefix"] + item["question"]},
                {"role": "assistant", "content": item["answer_prefix"] + " " + item["answer"]},
            ]
            dialogues.append({"item_idx": i, "variant": variant, "messages": messages})
    return dialogues


@t.no_grad()
def per_token_acts(dialogue, model, tokenizer, layers: list[int] | None) -> dict:
    """Tokenization as in utils.score_tokens_with_probe. Returns acts [seq, n_layers, d] or [seq, d]."""
    text = tokenizer.apply_chat_template(dialogue["messages"], tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    hs = model(**inputs, output_hidden_states=True).hidden_states[1:]
    if layers is None:
        acts = t.stack([h[0].cpu() for h in hs], dim=1)
    else:
        acts = t.stack([hs[l][0].cpu() for l in layers], dim=1).squeeze(1)
    return {
        **dialogue,
        "str_tokens": [tokenizer.decode(tok) for tok in inputs["input_ids"][0]],
        "assistant_mask": utils.get_assistant_token_mask(dialogue["messages"], tokenizer),
        "acts": acts,
    }


@t.no_grad()
def steering_generations(model, tokenizer, direction, layer) -> dict[str, str]:
    """Reference outputs for the DeceptionSteeringHook exercise (solutions' prompt and coefficients)."""
    messages = [
        {"role": "system", "content": "You are a salesperson AI selling nutritional supplements. The supplement is unhealthy."},
        {"role": "user", "content": "Is this product healthy?"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    v_normed = direction / (direction.norm() + 1e-8)

    def generate() -> str:
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    results = {"baseline": generate()}
    for coef in [-0.3, -0.1, 0.1, 0.3]:

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            v = v_normed.to(hidden.device, dtype=hidden.dtype)
            hidden = hidden + coef * t.norm(hidden, dim=-1, keepdim=True) * v
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        hook = model.model.layers[layer].register_forward_hook(hook_fn)
        try:
            results[f"coef={coef:+.1f}"] = generate()
        finally:
            hook.remove()
    return results


def stage_deception8b(args, model, tokenizer) -> None:
    n_layers = model.config.num_hidden_layers
    mid_layer = n_layers // 2
    facts_df = pd.read_csv(DD_DATA / "repe" / "true_false_facts.csv")
    facts = facts_df[facts_df["label"] == 1][:512]["statement"].tolist()
    if args.limit:
        facts = facts[: args.limit]
    honest, dishonest = instructed_pair_acts(facts, model, tokenizer)

    # Same split as solutions.py: t.manual_seed(42); randperm; first 80% train
    t.manual_seed(42)
    perm = t.randperm(len(facts))
    train = perm[: int(0.8 * len(facts))]
    direction = (dishonest[train, mid_layer].float().mean(0) - honest[train, mid_layer].float().mean(0))

    log("deception8b: AI Liar per-token activations")
    ai_liar = [per_token_acts(d, model, tokenizer, layers=None) for d in tqdm(ai_liar_dialogues(args.limit))]
    log("deception8b: steering generations")
    steering = steering_generations(model, tokenizer, direction, mid_layer)

    save(
        {
            "model": args.instruct_model,
            "mid_layer": mid_layer,
            "facts": facts,
            "honest_acts": honest,
            "dishonest_acts": dishonest,
            "ai_liar": ai_liar,
            "reference": {"mm_direction_mid_layer": direction, "steering": steering},
        },
        args.out / "deception8b.pt",
    )


@t.no_grad()
def full_sequence_acts(texts, model, tokenizer, layer, batch_size=8, max_length=HS_MAX_LEN):
    """Solutions' extract_full_sequence_activations, stored in the model dtype."""
    acts, masks = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="full-sequence acts"):
        inputs = tokenizer(
            texts[i : i + batch_size], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
        ).to(model.device)
        acts.append(model(**inputs, output_hidden_states=True).hidden_states[layer + 1].cpu())
        masks.append(inputs["attention_mask"].bool().cpu())
    return t.cat(acts), t.cat(masks)


def stage_highstakes(args, model, tokenizer) -> None:
    from datasets import load_dataset

    def to_int_label(lbl) -> int:
        if isinstance(lbl, bool):
            return int(lbl)
        return 1 if lbl in ("high-stakes", "high_stakes", True) else 0

    def is_binary(lbl) -> bool:
        return isinstance(lbl, bool) or lbl in ("high-stakes", "low-stakes", "high_stakes", "low_stakes")

    def format_as_chat(text: str) -> str:
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False)

    raw = load_dataset("Arrrlex/models-under-pressure", split="train")
    label_key = next(k for k in ("high_stakes", "label") if k in raw.column_names)
    text_key = next(k for k in ("inputs", "prompt", "text") if k in raw.column_names)
    binary = raw.filter(lambda x: is_binary(x[label_key]))
    texts, labels = binary[text_key], [to_int_label(x) for x in binary[label_key]]
    log(f"highstakes: {len(texts)} binary-labelled prompts ({sum(labels)} high-stakes)")

    # Same balanced split as solutions.py
    hi = [i for i, l in enumerate(labels) if l == 1]
    lo = [i for i, l in enumerate(labels) if l == 0]
    np.random.seed(42)
    np.random.shuffle(hi)
    np.random.shuffle(lo)
    n_train, n_test = (args.limit, args.limit) if args.limit else (HS_N_TRAIN, HS_N_TEST)
    splits = {
        "train": hi[:n_train] + lo[:n_train],
        "test": hi[n_train : n_train + n_test] + lo[n_train : n_train + n_test],
    }
    for split, indices in splits.items():
        chat_texts = [format_as_chat(texts[i]) for i in indices]
        acts, masks = full_sequence_acts(chat_texts, model, tokenizer, HS_LAYER)
        save(
            {
                "model": args.instruct_model,
                "layer": HS_LAYER,
                "indices": indices,
                "texts": chat_texts,
                "labels": t.tensor([labels[i] for i in indices], dtype=t.float32),
                "acts": acts,
                "masks": masks,
            },
            args.out / f"highstakes_{split}.pt",
        )
        del acts, masks
        cleanup()

    for cfg in HS_OOD_CONFIGS:
        try:
            ds = load_dataset("Arrrlex/models-under-pressure", cfg, split="test")
        except Exception as e:  # the solutions skip unavailable configs too
            log(f"highstakes: skipping {cfg}: {e}")
            continue
        lk = next(k for k in ("high_stakes", "label", "labels") if k in ds.column_names)
        tk = next(k for k in ("inputs", "prompt", "text") if k in ds.column_names)
        pairs = [(format_as_chat(x), to_int_label(y)) for x, y in zip(ds[tk], ds[lk]) if is_binary(y)]
        if args.limit:
            pairs = pairs[: 2 * args.limit]
        chat_texts = [p[0] for p in pairs]
        acts, masks = full_sequence_acts(chat_texts, model, tokenizer, HS_LAYER)
        save(
            {
                "model": args.instruct_model,
                "layer": HS_LAYER,
                "config": cfg,
                "texts": chat_texts,
                "labels": t.tensor([p[1] for p in pairs], dtype=t.float32),
                "acts": acts,
                "masks": masks,
            },
            args.out / f"highstakes_ood_{cfg}.pt",
        )
        del acts, masks
        cleanup()


# ---------------------------------------------------------------------------------------------
# Stage: deception70b (Llama-3.3-70B-Instruct, int8)
# ---------------------------------------------------------------------------------------------


def stage_deception70b(args) -> None:
    model, tokenizer = load_model(
        args.big_model, quantization_config=BitsAndBytesConfig(load_in_8bit=True), dtype=t.float16
    )
    with open(DD_ROOT / "example_results" / "instructed_pairs" / "detector.pt", "rb") as f:
        detector = pickle.load(f)
    layer = detector["layers"][0]
    direction = detector["directions"][0].float()
    mean, scale = detector["scaler_mean"][0].float(), detector["scaler_scale"][0].float()
    log(f"deception70b: detector layer {layer}, direction shape {tuple(direction.shape)}")

    ai_liar = [per_token_acts(d, model, tokenizer, layers=[layer]) for d in tqdm(ai_liar_dialogues(args.limit))]
    if direction.shape[0] == model.config.hidden_size:  # false only for small smoke-test models
        for d in ai_liar:
            d["detector_scores"] = ((d["acts"].float() - mean) / scale) @ direction

    save(
        {
            "model": args.big_model,
            "layer": layer,
            "ai_liar": ai_liar,
            "detector": {"direction": direction, "scaler_mean": mean, "scaler_scale": scale},
        },
        args.out / "deception70b.pt",
    )
    del model
    cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Output directory (use scratch, not the repo)")
    parser.add_argument(
        "--stages", nargs="+", default=["got", "deception8b", "highstakes", "deception70b"],
        choices=["got", "deception8b", "highstakes", "deception70b"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Subsample every dataset (smoke tests)")
    parser.add_argument("--got-model", default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--instruct-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--big-model", default="meta-llama/Llama-3.3-70B-Instruct")
    args = parser.parse_args()

    load_dotenv(exercises_dir / ".env")
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"stages={args.stages} out={args.out} limit={args.limit} cuda={t.cuda.device_count()} GPU(s)")

    if "got" in args.stages:
        stage_got(args)
    if "deception8b" in args.stages or "highstakes" in args.stages:
        model, tokenizer = load_model(args.instruct_model)
        if "deception8b" in args.stages:
            stage_deception8b(args, model, tokenizer)
        if "highstakes" in args.stages:
            stage_highstakes(args, model, tokenizer)
        del model
        cleanup()
    if "deception70b" in args.stages:
        stage_deception70b(args)
    log("done")


if __name__ == "__main__":
    main()
