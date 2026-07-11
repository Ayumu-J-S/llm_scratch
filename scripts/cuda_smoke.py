from __future__ import annotations

import argparse
import json
import os
import subprocess

import torch
import torch.nn.functional as F

from models.simple_decoder_transformer import SimpleDecoderTransformer


STEPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exactly ten BF16 CUDA optimizer steps")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def visible_compute_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}


def run_smoke() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requires an available CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA smoke requires BF16 support")

    torch.manual_seed(20260711)
    torch.cuda.manual_seed_all(20260711)
    device = torch.device("cuda")
    model = SimpleDecoderTransformer(
        vocab_size=257,
        embed_size=64,
        num_heads=4,
        max_len=32,
        num_layers=2,
        dropout=0.0,
        pad_token_id=None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, 257, (4, 32), device=device)
    labels = torch.randint(0, 257, (4, 32), device=device)
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("a model parameter is not on CUDA")
    if inputs.device.type != "cuda" or labels.device.type != "cuda":
        raise RuntimeError("smoke inputs and labels must be on CUDA")

    losses: list[float] = []
    output_dtype: str | None = None
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        output_dtype = str(logits.dtype)
        if logits.device.type != "cuda" or loss.device.type != "cuda":
            raise RuntimeError("smoke logits and loss must be on CUDA")
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite smoke loss")
        loss.backward()
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if any(parameter.grad is None for parameter in trainable_parameters):
            raise RuntimeError("a trainable parameter is missing its gradient")
        gradients = [parameter.grad for parameter in trainable_parameters]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("non-finite gradients")
        if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
            raise RuntimeError("all gradients are zero")
        optimizer.step()
        losses.append(float(loss.detach()))

    if output_dtype != "torch.bfloat16":
        raise RuntimeError(f"autocast output was not BF16: {output_dtype}")
    optimizer_cuda_tensors = [
        value
        for state in optimizer.state.values()
        for key, value in state.items()
        if key != "step" and isinstance(value, torch.Tensor)
    ]
    if not optimizer_cuda_tensors or any(
        value.device.type != "cuda" for value in optimizer_cuda_tensors
    ):
        raise RuntimeError("AdamW tensor state is not entirely on CUDA")

    torch.cuda.synchronize()
    current_pid = os.getpid()
    visible_pids = visible_compute_pids()
    if current_pid not in visible_pids:
        raise RuntimeError(
            f"current PID {current_pid} was not visible as an nvidia-smi compute process: "
            f"{sorted(visible_pids)}"
        )

    return {
        "steps": len(losses),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "autocast_output_dtype": output_dtype,
        "losses": losses,
        "finite_losses": all(torch.isfinite(torch.tensor(losses))),
        "optimizer_state_device": "cuda",
        "pid": current_pid,
        "pid_visible_in_nvidia_smi": True,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "memory_interpretation": (
            "CUDA allocator values are not total DGX Spark unified-memory use."
        ),
    }


def main() -> None:
    args = parse_args()
    report = run_smoke()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"PASS: {report['steps']} BF16 optimizer steps on {report['gpu_name']} "
            f"(PID {report['pid']} visible in nvidia-smi)"
        )


if __name__ == "__main__":
    main()
