from functools import wraps

# GPT2-XL hyperparameters
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400
dtype = 4 # float32

rope_theta = 10000


def parting_line(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        print(">" * 20 + "<" * 20)
        print(fun.__name__+" starts")
        print(">" * 20 + "<" * 20)
        print("=" * 40)
        fun(*args, **kwargs)
        print("=" * 40)
        print("\n\n")
    return wrapper

@parting_line
def problem_unicode1() -> None:
    print(chr(0))
    print(chr(0).__repr__())
    
    # run in Python interpreter
    # >>> chr(0)
    # >>> print(chr(0))
    # >>> "this is a test" + chr(0) + "string"
    # >>> print("this is a test" + chr(0) + "string")

@parting_line
def problem_unicode2() -> None:
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    try:
        print(decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8")))
    except Exception as e:
        print("Error in process `hello! こんにちは!`")
        print(f"{'hello! こんにちは!'.encode("utf-8")=}\n")
    
    try:
        print(bytes([255, 235]).decode("utf-8"))
    except Exception as e:
        print("Error in process `bytes([255, 235])`")
        print(e)

@parting_line
def problem_train_bpe_tinystories() -> None:
    # run with command line
    # python -m cprofile -o ./results/result_tiny.prof problems.py
    
    from cs336_basics.tokenizer_train import train_bpe, save_bpe
    vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    save_bpe(vocab, merges, "./results/", "tiny")

@parting_line
def problem_train_bpe_expts_owt() -> None:
    from cs336_basics.tokenizer_train import train_bpe, save_bpe
    vocab, merges = train_bpe("./data/OpenWebText_sample.txt", 32000, ["<|endoftext|>"])
    save_bpe(vocab, merges, "./results/", "owt")

@parting_line
def problem_train_bpe_tinystories_profile() -> None:
    import pstats

    p = pstats.Stats("./results/result_tiny.prof")
    p.strip_dirs().sort_stats("cumtime").print_stats(30)

@parting_line
def problem_transformer_accounting_trainable_parameters() -> None:
    """
    vocab_size: v
    context_length: s
    num_layers: l
    d_model: d
    num_heads: h
    d_ff: f
    
    trainable parameters: 
    
    TransformerLM: 2vd + l * (4d^2 + 3df + 2d) + d
        Token Embedding: vd
        RoPE: 0
        TransformerBlock: l * (4d^2 + 3df + 2d)
            Attention: 
                q, k, v, o: 4d^2
            Feed Forward: 3df
            2 * Layer Norm: 2d
            
        Final Layer Norm: d
        LM Head: vd
    """
    
    import torch
    from cs336_basics.model import TransformerLM

    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        dtype=torch.float32
    )

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Memory (float32): {total_params * 4 / (1024 ** 3):.2f} GB")

@parting_line
def problem_transformer_accounting_matrix_multiplication_flops() -> None:
    """
    vocab_size: v
    context_length: s
    num_layers: l
    d_model: d
    num_heads: h
    d_ff: f
    
    flops: 
    
    TransformerLM: 2bsdv + bl * (8sd^2 + 4s^2d + 6sdf)
        Token Embedding: 0
        RoPE: 0
        TransformerBlock: bl * (8sd^2 + 4s^2d + 6sdf)
            Attention: 
                q, k, v:
                    3 * bsd @ dd: 3 * 2 * bsd^2
                attention:
                    bhs(d/h) @ bhs(d/h) -> bhss: 2 * bh * s * (d/h) * s = 2 * bs^2d
                attention @ v: 
                    bhss @ bhs(d/h) -> bhs(d/h): 2 * bh * s * s * (d/h) = 2 * bs^2d
                o:
                    bsd @ dd: 2 * bsd^2
            Feed Forward: 6bsdf
                2 * bsd @ df -> 2 * 2 * bsdf
                bsf @ fd -> 2 * bsdf
            2 * Layer Norm: 0
        Final Layer Norm: 0
        LM Head: 2bsdv
    """
    import torch
    from cs336_basics.model import TransformerLM
    from fvcore.nn import FlopCountAnalysis

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        dtype=torch.float32,
    )
    model.eval()

    input_tokens = torch.randint(0, vocab_size, (1, context_length), dtype=torch.long)
    flops = FlopCountAnalysis(model, input_tokens)
    flops_per_op = flops.by_operator()
    matmul_ops = {k: v for k, v in flops_per_op.items() if "matmul" in k.lower() or "einsum" in k.lower()}
    total_matmul_flops = sum(matmul_ops.values())
    print(f"Total MatMul-related FLOPs: {total_matmul_flops * 2:,.0f}")

    import re

    module_flops = flops.by_module()
    matmul_modules = {name: flop for name, flop in module_flops.items() if re.search(r"(q_proj|k_proj|v_proj|output_proj|w1|w2|w3|lm_head)", name)}
    for mod, flop in matmul_modules.items():
        print(f"{mod:50s}: {flop * 2:,.0f}")


@parting_line
def problem_transformer_accounting_other_size_model() -> None:
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    def compute_gpt2_flops_and_params(
        model_name: str, v: int, s: int, l: int, d: int, h: int, f: int, b: int = 1
    ) -> pd.DataFrame:
        flops_embed = 0
        flops_rope = 0
        flops_lm_head = 2 * b * s * d * v
        flops_attention_proj = 8 * b * s * d**2
        flops_attention_scores = 4 * b * s**2 * d
        flops_ffn = 6 * b * s * d * f
        flops_per_layer = flops_attention_proj + flops_attention_scores + flops_ffn
        total_flops = flops_embed + flops_rope + flops_lm_head + l * flops_per_layer

        params_embed = v * d
        params_lm_head = v * d
        params_attention_proj = 4 * d**2
        params_ffn = 3 * d * f
        params_ln = 2 * d
        params_per_layer = params_attention_proj + params_ffn + params_ln
        params_final_ln = d
        total_params = params_embed + params_lm_head + l * params_per_layer + params_final_ln

        rows = [
            {"Model": model_name, "Component": "Embedding", "FLOPs": flops_embed, "Params": params_embed},
            {"Model": model_name, "Component": "Attention Projections", "FLOPs": l * flops_attention_proj, "Params": l * params_attention_proj},
            {"Model": model_name, "Component": "Attention Softmax+Scores", "FLOPs": l * flops_attention_scores, "Params": 0},
            {"Model": model_name, "Component": "Feed Forward", "FLOPs": l * flops_ffn, "Params": l * params_ffn},
            {"Model": model_name, "Component": "LayerNorm", "FLOPs": 0, "Params": l * params_ln + params_final_ln},
            {"Model": model_name, "Component": "LM Head", "FLOPs": flops_lm_head, "Params": params_lm_head},
        ]
        df = pd.DataFrame(rows)
        df["FLOPs %"] = df["FLOPs"] / total_flops * 100
        df["Params %"] = df["Params"] / total_params * 100
        return df

    # Model config
    models = [
        ("GPT-2 Small", 50257, 1024, 12, 768, 12, 3072),
        ("GPT-2 Medium", 50257, 1024, 24, 1024, 16, 4096),
        ("GPT-2 Large", 50257, 1024, 36, 1280, 20, 5120),
        ("GPT-2 XL", 50257, 1024, 48, 1600, 25, 6400),
        ("GPT-2 XL (16384 ctx)", 50257, 16384, 48, 1600, 25, 6400),
    ]

    all_dfs = [compute_gpt2_flops_and_params(*args) for args in models]
    all_df = pd.concat(all_dfs, ignore_index=True)
    print(all_df.to_string(index=False))

    # Plotting
    os.makedirs("results", exist_ok=True)
    model_order = [m[0] for m in models]
    components = all_df["Component"].unique()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    metrics = ["FLOPs %", "Params %"]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for comp in components:
            ys = []
            for model in model_order:
                val = all_df[(all_df["Model"] == model) & (all_df["Component"] == comp)][metric].values
                ys.append(val[0] if len(val) > 0 else 0)
            ax.plot(model_order, ys, marker='o', label=comp)

        ax.set_title(f"Component-wise {metric} Across GPT-2 Models")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False)

    fig.suptitle("Component FLOPs % and Params % Across GPT-2 Model Sizes", fontsize=16)
    save_path = "results/component_percent_dual_plot.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

@parting_line
def problem_learning_rate_tuning() -> None:
    import torch
    import matplotlib.pyplot as plt
    from cs336_basics.optimizer import SGD  # 假设你已实现自定义SGD

    def run_experiment(learning_rate: float, steps: int = 10):
        losses = []
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=learning_rate)
        for t in range(steps):
            opt.zero_grad()
            loss = (weights**2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    # 学习率设置
    learning_rates = [1, 1e1, 1e2, 1e3]
    results = {f"lr={lr:g}": run_experiment(lr) for lr in learning_rates}

    # 绘图
    plt.figure(figsize=(10, 6))
    for label, losses in results.items():
        plt.plot(range(1, len(losses)+1), losses, label=label)

    plt.title("Loss vs Training Step for Different Learning Rates")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.yscale("log")  # 对数尺度观察更清楚
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("results/learning_rate_tuning.png")
    print("Saved: results/learning_rate_tuning.png")

@parting_line
def problem_adamwAccounting(vocab_size: int = vocab_size, context_length: int = context_length, num_layers: int = num_layers, d_model: int = d_model, num_heads: int = num_heads, d_ff: int = d_ff, MAX_MEMORY: int = 80) -> None:
    """
    vocab_size: v
    context_length: s
    num_layers: l
    d_model: d
    num_heads: h
    d_ff: f
    
    Memory(AdamW): 
    Parameters = Trainable parameters + Non-trainable parameters = [2vd + l * (4d^2 + 3df + 2d) + d] + sd/h (RoPE)
    gradients = Trainable parameters = 2vd + l * (4d^2 + 3df + 2d) + d
    optimizer state = 2 * Trainable parameters = 2[2vd + l * (4d^2 + 3df + 2d) + d]
    Activations: 2bsd + bsv + l * (6bsd + 3bs^2h + 3bsf)
        Embedding: bsd
        RoPE: 0
        TransformerBlock: l * (6bsd + 3bs^2h+ 3bsf)
            Attention: 4bsd+3bhs^2
                q, k, v: 3 * bsd
                Q^TK: bhs^2
                softmax: bhs^2
                attention * v: bhs^2
                output: bsd
            Feed Forward: 3bsf
                W_1x: bsf
                W_3x: bsf
                Swish: bsf
            2 * Layer Norm: 2bsd
            
        Final Layer Norm: bsd
        LM Head: bsv
    
    Total = [2vd + l * (4d^2 + 3df + 2d) + d] + sd/h + 3[2vd + l * (4d^2 + 3df + 2d) + d] + 2bsd + bsv + l * (6bsd + 3bs^2h + 3bsf)
    = 8vd + 2bsd + bsv + 4l(4d^2 + 3df + 2d) + l(6bsd + 3bs^2 + 3bsf) + 4d + sd/h
    = b(2sd + sv + l(6sd + 3s^2h + 3sf)) + 4d + sd/h + 8vd + 4l(4d^2 + 3df + 2d)
    """
    
    s = context_length
    d = d_model
    v = vocab_size
    l = num_layers
    h = num_heads
    f = d_ff
    
    term1 = (2 * s * d + s * v + l * (6 * s * d + 3 * s**2 * h + 3 * s * f))
    term2 = 4 * d + s * d / h + 8 * v * d + 4 * l * (4 * d**2 + 3 * d * f + 2 * d)

    total = f"{term1} * b + {term2}"
    print(f"Total (AdamW): {total}")
    
    term1_memory = term1 * dtype / 1024 ** 3
    term2_memory = term2 * dtype / 1024 ** 3
    total_memory = f"{term1_memory} * b + {term2_memory}"
    print(f"Memory (AdamW): {total_memory} GB")
    
    print(f"Max Batch Size: {(MAX_MEMORY - term2_memory) / term1_memory}")
    
@parting_line
def problem_adamwAccounting_onestep_adamw_flops():
    """
    Trainable Parameters: θ
    
    m: 3θ
    v: 4θ
    θ_1: 5θ
    θ_2: θ
    total: 13θ
    """
    pass

@parting_line
def problem_adamwAccounting_train_time():
    """
    There are certain issues with the question setting. Here, we do not consider the memory of the A100, but only its FLOP/s.
    
    total flops: 6bsdv + 3bl(8sd^2 + 4s^2d + 6sdf) + 26vd + 13d + 13l(4d^2 + 3df + 2d)
        forward: F = 2bsdv + bl * (8sd^2 + 4s^2d + 6sdf)
        backward: 2F = 4bsdv + 2bl * (8sd^2 + 4s^2d + 6sdf)
        adamw: 13θ = 13 * (2vd + l * (4d^2 + 3df + 2d) + d)
    """
    
    b = batch_size = 1024
    steps = num_steps = 400e3
    MFU = 0.5
    peak_flops_TFLOPs = 19.5e12
    
    
    
    v = vocab_size
    s = context_length
    l = num_layers
    d = d_model
    f = d_ff

    # Forward pass FLOPs
    forward_flops = 2 * b * s * d * v + b * l * (8 * s * d**2 + 4 * s**2 * d + 6 * s * d * f)
    
    # Backward pass FLOPs (2x forward)
    backward_flops = 2 * forward_flops

    # AdamW optimizer FLOPs
    adamw_flops = 13 * (2 * v * d + l * (4 * d**2 + 3 * d * f + 2 * d) + d)

    # Total per-step FLOPs
    total_flops_per_step = forward_flops + backward_flops + adamw_flops

    # Total FLOPs for all training steps
    total_training_flops = total_flops_per_step * steps

    # Effective hardware FLOP throughput
    effective_flops_per_sec = MFU * peak_flops_TFLOPs

    # Compute time in seconds and convert to days
    training_time_seconds = total_training_flops / effective_flops_per_sec
    training_time_days = training_time_seconds / (60 * 60 * 24)

    print(f"{training_time_days} days")
    
    
    
if __name__ == '__main__':
    problem_adamwAccounting(10000, 128, 12, 768, 12, 3072, 8)