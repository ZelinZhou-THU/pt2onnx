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
        --output models/best.onnx \\
        --converter yolov8_seg
"""

import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _discover_converters() -> dict:
    """Return {name: CONVERTER_META} for every `*_converter.py` in this dir.

    Loads each module via importlib so its CONVERTER_META is parsed, but
    does NOT instantiate the converter class. This stays cheap because the
    converter classes import torch/ultralytics lazily inside their methods.

    The discovered meta dict gains a private ``_module_file`` key containing
    the absolute path of the source file. ``_resolve_converter_class`` uses
    this to avoid reconstructing the path from the converter name, which
    would break when the file name differs from the converter ``name`` field
    (e.g. ``yolo_converter.py`` exposes ``name="yolov8_seg"``).
    """
    result = {}
    converter_dir = Path(__file__).parent
    for py_file in sorted(converter_dir.glob("*_converter.py")):
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            meta = getattr(mod, "CONVERTER_META", None)
            if meta and "name" in meta:
                meta["_module_file"] = str(py_file)
                result[meta["name"]] = meta
        except Exception as exc:
            logging.getLogger("pt2onnx").warning("Skipping %s: %s", py_file.name, exc)
    return result


def _resolve_converter_class(name: str, meta: dict):
    """Load the converter module on demand and return its class.

    Uses CONVERTER_META['class_name'] when present; otherwise falls back to
    scanning for any class whose name ends with 'Converter'.

    Prefers the ``_module_file`` path stored by ``_discover_converters`` over
    reconstructing the path from the converter name.
    """
    module_file = meta.get("_module_file")
    if module_file:
        module_path = Path(module_file)
    else:
        # Fallback: reconstruct from naming convention {name}_converter.py
        converter_dir = Path(__file__).parent
        module_name = f"{name.replace('-', '_')}_converter"
        module_path = converter_dir / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Converter module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class_name = meta.get("class_name")
    if class_name and hasattr(mod, class_name):
        return getattr(mod, class_name)

    # Fallback: pick the first class whose name ends with 'Converter'.
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and attr_name.endswith("Converter"):
            return attr
    raise AttributeError(
        f"No converter class found in {module_path} "
        f"(CONVERTER_META.class_name={class_name!r}, scan for *Converter failed)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert model checkpoint to ONNX (.onnx + .meta.json sidecar)."
    )
    parser.add_argument(
        "--converter",
        default="yolov8_seg",
        help="Converter name (default: yolov8_seg). Use --list-converters to see available.",
    )
    parser.add_argument(
        "--list-converters",
        action="store_true",
        help="List all available converters and exit.",
    )
    parser.add_argument("--input", help="Path to input model checkpoint (required unless --list-converters)")
    parser.add_argument("--output", help="Output .onnx file path (required unless --list-converters)")
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

    converters = _discover_converters()

    if args.list_converters:
        for name, meta in converters.items():
            deps = ", ".join(meta.get("dependencies", []))
            print(
                f"  {name}: {meta.get('display_name', name)} "
                f"[deps: {deps}]\n    {meta.get('description', '')}"
            )
        return

    # Validate input/output for the actual conversion path
    if not args.input or not args.output:
        parser.error("--input and --output are required (omit --list-converters)")

    if args.converter not in converters:
        print(
            f"Unknown converter: {args.converter!r}. "
            f"Available: {sorted(converters.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    meta = converters[args.converter]
    try:
        converter_class = _resolve_converter_class(args.converter, meta)
    except (FileNotFoundError, AttributeError) as exc:
        logging.getLogger("pt2onnx").error("%s", exc)
        sys.exit(1)

    try:
        converter = converter_class()
        result_meta = converter.convert(
            pt_path=args.input,
            onnx_path=args.output,
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
        )
    except Exception as exc:
        logging.getLogger("pt2onnx").error("Conversion failed: %s", exc, exc_info=True)
        sys.exit(1)

    nc = result_meta.get("nc", "?")
    imgsz = result_meta.get("imgsz", "?")
    opset = result_meta.get("opset", "?")
    print(
        f"✓ Export succeeded via {args.converter}: nc={nc}, imgsz={imgsz}, opset={opset}\n"
        f"  Output:  {args.output}\n"
        f"  Metadata: {args.output}.meta.json"
    )


if __name__ == "__main__":
    main()
