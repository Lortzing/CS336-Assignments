import torch
from cs336_basics.model import softmax

def top_p_sampling(probs: torch.Tensor, p: float) -> int:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Select smallest subset of tokens whose cumulative probability exceeds p
    cutoff = torch.searchsorted(cumulative_probs, p)
    top_indices: torch.LongTensor = sorted_indices[:cutoff + 1]
    top_probs = probs[top_indices]
    top_probs = top_probs / top_probs.sum()  # renormalize

    sampled_idx = torch.multinomial(top_probs, num_samples=1)
    return top_indices[sampled_idx].item()


@torch.no_grad()
def decoding(
    model,
    prompt: list[int],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
) -> str:
    model.eval()
    generated = list(prompt)
    x = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        if x.size(1) > model.context_length:
            x = x[:, -model.context_length:]  # truncate left if too long

        logits = model(x)
        next_token_logits = logits[0, -1, :]

        next_token_logits = next_token_logits / temperature

        probs = softmax(next_token_logits, dim=-1)

        next_token = top_p_sampling(probs, top_p)

        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        x = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = tokenizer.decode(generated)

    return generated

if __name__ == '__main__':
    from cs336_basics.model import TransformerLM
    from cs336_basics.tokenizer import Tokenizer
    model = TransformerLM(
        vocab_size=10000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        context_length=128,
        d_ff=3072,
        rope_theta=10000,
    ).to("cuda")
    model.load_state_dict(torch.load("./checkpoints/step_3500.pt", map_location="cuda:0", weights_only=True)["model"])
    tokenizer = Tokenizer.from_files(vocab_filepath="./results/vocab_tiny_gpt2.json", merges_filepath="./results/merges_tiny_gpt2.txt")
    prompt = tokenizer.encode("hello,")
    print(decoding(model, prompt, device="cuda:0"))