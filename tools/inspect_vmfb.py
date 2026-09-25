"""Inspect input/output signatures and shapes of Torq VMFB models."""

import os
import sys
import numpy as np

try:
    from torq.runtime import VMFBInferenceRunner
except ImportError:
    from torq_runtime import VMFBInferenceRunner


def inspect_vmfb(model_path: str):
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    print(f"\n=======================================================")
    print(f"Model: {model_path}")
    print(f"Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"=======================================================")

    runner = VMFBInferenceRunner(model_path)
    
    # Check attributes on runner
    for attr in ["function_names", "inputs_info", "outputs_info"]:
        if hasattr(runner, attr):
            print(f"{attr}: {getattr(runner, attr)}")

    # Check underlying IREE VM function signature
    if hasattr(runner, "_invoker"):
        invoker = runner._invoker
        print(f"Invoker: {invoker}")
        for attr in ["function", "_function", "_vm_function"]:
            if hasattr(invoker, attr):
                fn = getattr(invoker, attr)
                print(f"Function ({attr}): {fn}")
                if hasattr(fn, "signature"):
                    print(f"Signature: {fn.signature}")

    # Inspect token embeddings file if exists
    emb_path = "models/decoder_token_embeddings.npy"
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        print(f"\nToken embeddings: shape={emb.shape}, dtype={emb.dtype}")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["models/encoder.vmfb", "models/decoder.vmfb"]
    for m in models:
        inspect_vmfb(m)
