"""BC fine-tune: prompt -> action pairs to LoRA adapter. ROCm-ready.

Same code runs on MI300X (ROCm PyTorch) and NVIDIA/CPU — device auto-detects.
`torch.cuda` APIs route to ROCm via HIP, no code change needed.

Cloud (MI300X, Quick Start PyTorch image, inside container):
  docker exec -it rocm bash
  git clone https://github.com/singharyan44/amd-grid-learner && cd amd-grid-learner
  pip install -r requirements.txt
  python data/build_bc_dataset.py --expert --stage S1 --episodes 500 --out data/bc_expert_s1.json
  python train/bc_train.py --data data/bc_expert_s1.json --model Qwen/Qwen2-0.5B-Instruct --out models/bc_s1 --epochs 3

Local smoke (no download, tiny random model, CPU, seconds):
  python train/bc_train.py --smoke --data data/bc_expert_s1_50.json --out models/smoke
"""
import argparse, json, os, sys


def load_pairs(path, limit=None):
    with open(path) as f:
        data = json.load(f)
    pairs = []
    for ep in data:
        for step in ep["trace"]:
            pairs.append((step["prompt"], step["action"]))
            if limit and len(pairs) >= limit:
                return pairs
    return pairs


def build_texts(pairs):
    # completion-style: loss masked to the action token(s) only (see mask_labels)
    return [(p, a) for p, a in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bc_expert_s1_50.json")
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--out", default="models/bc_s1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random GPT-2 on CPU, 20 pairs, 1 epoch — no downloads")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"  # cuda == ROCm HIP on AMD
    print(f"torch={torch.__version__} device={device} cuda_avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    pairs = load_pairs(a.data, limit=20 if a.smoke else None)
    print(f"pairs={len(pairs)}")
    texts = build_texts(pairs)

    if a.smoke:
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=tok.vocab_size,
                         n_positions=256, n_ctx=256)
        model = GPT2LMHeadModel(cfg)
        max_len, epochs = 128, 1
    else:
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True)
        max_len, epochs = a.max_len, a.epochs

    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj", "c_attn"],
                      task_type="CAUSAL_LM")
    try:
        model = get_peft_model(model, lora)
    except ValueError as e:
        # tiny smoke model lacks some targets — fall back to attn-only
        print(f"lora target fallback: {e}")
        lora.target_modules = ["c_attn"]
        model = get_peft_model(model, lora)
    model.to(device).train()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    act_ids = {act: tok.encode(" " + act, add_special_tokens=False) for act in
               ["up", "down", "left", "right", "wait"]}

    def batchify(i):
        chunk = texts[i:i + a.batch]
        enc = tok([p for p, _ in chunk], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        # mask prompt: loss only on trailing action tokens
        for r, (_, act) in enumerate(chunk):
            ids = act_ids[act]
            L = int(attn[r].sum())
            labels[r, :max(0, L - len(ids))] = -100
        return input_ids, attn, labels

    step, losses = 0, []
    for ep in range(epochs):
        for i in range(0, len(texts), a.batch):
            input_ids, attn, labels = batchify(i)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            step += 1
            losses.append(float(out.loss))
            if step % 10 == 0 or a.smoke:
                print(f"ep={ep} step={step} loss={losses[-1]:.4f}")

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump({"pairs": len(pairs), "epochs": epochs, "steps": step,
                   "loss_first": losses[0] if losses else None,
                   "loss_last": losses[-1] if losses else None,
                   "torch": torch.__version__, "device": device,
                   "model": "smoke-gpt2" if a.smoke else a.model}, f, indent=2)
    print(f"saved -> {a.out} loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
