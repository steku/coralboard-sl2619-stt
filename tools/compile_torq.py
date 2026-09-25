"""Helper script to compile ASR ONNX / TOSA models to Torq VMFB artifacts for Coralboard SL2619 NPU."""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger(__name__)


def check_torq_compiler() -> bool:
    """Check if torq-compiler or torq-compile command line tool is present."""
    for cmd in ["torq-compile", "iree-compile"]:
        try:
            res = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if res.returncode == 0:
                _LOGGER.info("Found compiler: %s (%s)", cmd, res.stdout.strip())
                return True
        except FileNotFoundError:
            pass
    return False


def compile_onnx_to_vmfb(
    onnx_path: str,
    output_vmfb_path: str,
    target_soc: str = "sl2610",
    compiler_opt: str = "aggressive",
) -> None:
    """Compile ONNX model to Torq NPU .vmfb artifact."""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Input model not found: {onnx_path}")

    _LOGGER.info("Starting compilation of %s -> %s", onnx_path, output_vmfb_path)
    _LOGGER.info("Target SoC: %s (Coralboard SL2619)", target_soc)

    compile_cmd = [
        "torq-compile",
        onnx_path,
        "-o", output_vmfb_path,
        f"--torq-target-soc={target_soc}",
        f"--opt-level={compiler_opt}",
    ]

    _LOGGER.info("Executing command: %s", " ".join(compile_cmd))
    try:
        proc = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        _LOGGER.info("Compilation successful! Output saved to: %s", output_vmfb_path)
    except FileNotFoundError:
        _LOGGER.error(
            "Compiler tool `torq-compile` not found in PATH!\n"
            "To install the Torq Compiler:\n"
            "  pip install https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_compiler-2.1.0-cp312-cp312-manylinux_2_28_x86_64.whl\n"
        )
        raise
    except subprocess.CalledProcessError as e:
        _LOGGER.error("Compilation failed with error:\n%s\n%s", e.stdout, e.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Compile ONNX model to Torq VMFB for Coralboard SL2619 NPU"
    )
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--output", default="encode.vmfb", help="Output .vmfb file path")
    parser.add_argument("--soc", default="sl2610", help="Target Synaptics SoC (sl2610 / sl2619)")
    args = parser.parse_args()

    try:
        compile_onnx_to_vmfb(args.onnx, args.output, target_soc=args.soc)
    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
