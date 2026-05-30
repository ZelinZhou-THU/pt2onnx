#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 pt2onnx Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
pt2onnx converter — YOLO .pt → ONNX

Run as a standalone process; do NOT import this module from other applications.

Usage:
    python cli.py \\
        --input  models/best.pt \\
        --output models/best.onnx
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YOLO .pt checkpoint to ONNX (.onnx + .meta.json sidecar)."
    )
    parser.add_argument("--input", required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx file path")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input resolution in pixels (default: 640)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Export with symbolic batch axis for batched inference (default: enabled)",
    )
    parser.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        help="Fix all dimensions (batch=1) for environments without dynamic axis support",
    )
    args = parser.parse_args()

    # Make yolo_converter importable regardless of cwd
    sys.path.insert(0, str(Path(__file__).parent))
    from yolo_converter import YoloConverter

    try:
        converter = YoloConverter()
        meta = converter.convert(
            pt_path=args.input,
            onnx_path=args.output,
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
        )
    except Exception as exc:
        logging.getLogger("pt2onnx").error("Conversion failed: %s", exc, exc_info=True)
        sys.exit(1)

    print(
        f"✓ Export succeeded: nc={meta['nc']}, imgsz={meta['imgsz']}, opset={meta['opset']}\n"
        f"  Output:  {args.output}\n"
        f"  Metadata: {args.output}.meta.json"
    )


if __name__ == "__main__":
    main()
