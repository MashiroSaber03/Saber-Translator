"""Standalone entrypoint for the pinned Transformers environment."""

import contextlib
import json
import os
import queue
import sys
import threading


def load_model(directory, weight_bytes):
    import torch
    import transformers
    from transformers import AutoModel, AutoProcessor

    if transformers.__version__ != "5.17.0":
        raise RuntimeError("unexpected Transformers version")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    requested = os.environ.get("SABER_VALEN_DEVICE", "auto")
    if requested == "auto":
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                and torch.cuda.mem_get_info()[0] > weight_bytes + 768 * 1024 ** 2):
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested not in ("cuda", "mps", "cpu"):
        raise RuntimeError("invalid Valen device")
    # Follow the pinned release on CPU/CUDA. MPS uses the verified BF16
    # backbone/FP32 head path; the original head defines its own precision.
    model = AutoModel.from_pretrained(
        directory, trust_remote_code=True, local_files_only=True, use_safetensors=True,
        dtype=torch.bfloat16 if requested == "mps" else "auto", attn_implementation="sdpa",
    ).to(requested).eval()
    processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    if {parameter.dtype for parameter in model.head.parameters()} != {torch.float32}:
        raise RuntimeError("Valen head did not load in FP32")
    storage = str(next(model.backbone.parameters()).dtype).removeprefix("torch.")
    compute = str(model.backbone_autocast_dtype).removeprefix("torch.")
    return model, processor, {
        "device": str(next(model.backbone.parameters()).device),
        "precision": f"weights={storage},backbone={compute},head=float32",
        "torch": torch.__version__,
    }


def main():
    commands = queue.Queue()

    def read():
        for line in sys.stdin:
            commands.put(line)
        # Also release the model if the supervisor dies during a forward pass.
        os._exit(0)

    pipe = sys.stdout

    def send(payload):
        print(json.dumps(payload, ensure_ascii=True, allow_nan=False), file=pipe, flush=True)

    with contextlib.redirect_stdout(sys.stderr):
        try:
            model, processor, info = load_model(sys.argv[1], int(sys.argv[2]))
            # Windows numerical runtimes initialize their native libraries
            # before the child starts a blocking input-reader thread.
            threading.Thread(target=read, daemon=True).start()
            send(info)
            while True:
                request = json.loads(commands.get())
                # The pinned release chooses its trained native execution mode.
                # Saber sends only one question with its own image state.
                result = model.predict(request, processor=processor)
                send(result)
        except Exception as error:
            text = str(error).lower()
            if isinstance(error, MemoryError) or "out of memory" in text or "not enough memory" in text:
                send({"error": "Valen 运行内存不足，请释放内存或选择较小的模型"})
            else:
                send({"error": f"Valen 原生推理失败（{type(error).__name__}），请校验模型及运行环境"})
            raise


if __name__ == "__main__":
    main()
