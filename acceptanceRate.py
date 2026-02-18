import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Example models (replace with yours)
drafter_name = "facebook/opt-125m"
verifier_name = "facebook/opt-350m"

tokenizer = AutoTokenizer.from_pretrained(verifier_name)

drafter = AutoModelForCausalLM.from_pretrained(
    drafter_name, torch_dtype=torch.float16
).to(device).eval()

verifier = AutoModelForCausalLM.from_pretrained(
    verifier_name, torch_dtype=torch.float16
).to(device).eval()


def sample_next(model, input_ids):
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1]
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, 1)
    return token


def speculative_acceptance(prompt, draft_len=5, steps=20):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    accepted = 0
    drafted = 0

    for _ in range(steps):
        # ---- Draft tokens ----
        draft_tokens = []
        cur = input_ids.clone()

        for _ in range(draft_len):
            tok = sample_next(drafter, cur)
            draft_tokens.append(tok)
            cur = torch.cat([cur, tok], dim=1)

        drafted += len(draft_tokens)

        # ---- Verify tokens sequentially ----
        for tok in draft_tokens:
            vtok = sample_next(verifier, input_ids)

            if vtok.item() == tok.item():
                accepted += 1
                input_ids = torch.cat([input_ids, tok], dim=1)
            else:
                # rejection → replace with verifier token
                input_ids = torch.cat([input_ids, vtok], dim=1)
                break

    return accepted / drafted


rate = speculative_acceptance("The future of AI is")
print("Acceptance rate:", rate)