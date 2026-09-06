"""Download WeDetect-Base-Uni and export its visual proposal model to ONNX.

The generated file defaults to:
    ~/.cache/wedetect/wedetect_anything_base.onnx

The backend discovers that path automatically.  This script keeps the GPL-v3
WeDetect source/checkpoint outside this repository and records its upstream
revision through the cloned checkout.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


REPOSITORY = "https://github.com/WeChatCV/WeDetect.git"
HF_REPOSITORY = "fushh7/WeDetect"
CHECKPOINT_NAME = "wedetect_base_uni.pth"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    env = os.environ.copy()
    # PyTorch's modern ONNX exporter prints Unicode status glyphs.  Windows
    # consoles commonly default to cp1252, which otherwise aborts a successful
    # graph capture with UnicodeEncodeError before the model is written.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "wedetect",
        help="destination for the upstream checkout, checkpoint and ONNX model",
    )
    parser.add_argument("--device", default="cpu", help="device used only during ONNX export")
    parser.add_argument("--refresh-source", action="store_true", help="re-clone the upstream repository")
    args = parser.parse_args()

    cache_dir = args.cache_dir.expanduser().resolve()
    source_dir = cache_dir / "source"
    output_path = cache_dir / "wedetect_anything_base.onnx"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_source and source_dir.exists():
        shutil.rmtree(source_dir)
    if not source_dir.exists():
        run(["git", "clone", "--depth", "1", REPOSITORY, str(source_dir)])

    checkpoint = Path(
        hf_hub_download(
            repo_id=HF_REPOSITORY,
            filename=CHECKPOINT_NAME,
            cache_dir=str(cache_dir / "huggingface"),
        )
    )

    export_script = source_dir / "wedetect_anything" / "export_onnx.py"
    run(
        [
            sys.executable,
            str(export_script),
            "--variant",
            "base",
            "--checkpoint",
            str(checkpoint),
            "--img-size",
            "640",
            "--device",
            args.device,
            "--output-dir",
            str(cache_dir),
        ],
        cwd=source_dir,
    )

    if not output_path.is_file():
        raise RuntimeError(f"Export finished without producing {output_path}")
    print(f"\nReady: {output_path}")
    print("The backend will discover this default path automatically.")


if __name__ == "__main__":
    main()
