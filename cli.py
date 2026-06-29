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
        --converter yolo_seg
"""

import argparse
import ast
import importlib.util
import logging
import sys
from pathlib import Path

# TODO: consolidate _discover_converters with converter_registry._scan_directory_for_converters
# They share AST-based CONVERTER_META extraction but diverge in error tolerance.
# Unify when adding the next converter.

# Force UTF-8 stdout/stderr — torch.onnx prints emoji (e.g. \u2705) that
# GBK (Windows zh-CN default locale) cannot encode, causing UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _discover_converters() -> dict:
    """Return {name: {"meta": CONVERTER_META, "file": Path}} for every `*_converter.py`.

    Uses AST parsing (``ast.literal_eval``) to read ``CONVERTER_META`` without
    executing the module. This is the structural guarantee that AGPL imports
    inside converter modules are never triggered during discovery — even if a
    future converter author places ``import ultralytics`` at the module top
    level, discovery stays clean.

    The public ``CONVERTER_META`` dict is kept unmodified; the file path is
    stored separately in a wrapper dict so internal state does not leak into
    the shared plugin protocol.
    """
    result = {}
    converter_dir = Path(__file__).parent
    for py_file in sorted(converter_dir.glob("*_converter.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "CONVERTER_META"
                ):
                    meta = ast.literal_eval(node.value)
                    if isinstance(meta, dict) and "name" in meta:
                        result[meta["name"]] = {"meta": meta, "file": py_file}
                    break
        except Exception as exc:
            logging.getLogger("pt2onnx").warning("Skipping %s: %s", py_file.name, exc)
    return result


def _resolve_converter_class(name: str, meta: dict, module_file: Path):
    """Load the converter module on demand and return its class.

    ``exec_module`` is called here — only when an actual conversion is
    requested — so AGPL imports happen in the subprocess that already
    expects them, not during discovery.

    Uses ``CONVERTER_META['class_name']`` when present; otherwise falls back
    to scanning for any class whose name ends with 'Converter'.
    """
    if not module_file.exists():
        raise FileNotFoundError(f"Converter module not found: {module_file}")
    spec = importlib.util.spec_from_file_location(module_file.stem, module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class_name = meta.get("class_name")
    if class_name and hasattr(mod, class_name):
        return getattr(mod, class_name)

    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and attr_name.endswith("Converter"):
            return attr
    raise AttributeError(
        f"No converter class found in {module_file} "
        f"(CONVERTER_META.class_name={class_name!r}, scan for *Converter failed)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert model checkpoint to ONNX (.onnx + .meta.json sidecar)."
    )
    parser.add_argument(
        "--converter",
        default="yolo_seg",
        help="Converter name (default: yolo_seg). Use --list-converters to see available.",
    )
    parser.add_argument(
        "--list-converters",
        action="store_true",
        help="List all available converters and exit.",
    )
    parser.add_argument("--input", help="Path to input model checkpoint (required unless --list-converters)")
    parser.add_argument("--output", help="Output .onnx file path (required unless --list-converters)")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input resolution in pixels (default: 640)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18)")
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export with symbolic batch axis (default: enabled). Use --no-dynamic to fix all dims.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Model type (SAM only; ignored by other converters). "
             "Example: vit_b / vit_l / vit_h.",
    )
    args = parser.parse_args()

    entries = _discover_converters()

    if args.list_converters:
        for name, entry in entries.items():
            converter_meta = entry["meta"]
            deps = ", ".join(converter_meta.get("dependencies", []))
            print(
                f"  {name}: {converter_meta.get('display_name', name)} "
                f"[deps: {deps}]\n    {converter_meta.get('description', '')}"
            )
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required (omit --list-converters)")

    if args.converter not in entries:
        print(
            f"Unknown converter: {args.converter!r}. "
            f"Available: {sorted(entries.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    entry = entries[args.converter]
    converter_meta = entry["meta"]
    module_file = entry["file"]
    try:
        converter_class = _resolve_converter_class(args.converter, converter_meta, module_file)
    except (FileNotFoundError, AttributeError) as exc:
        logging.getLogger("pt2onnx").error("%s", exc)
        sys.exit(1)

    try:
        converter = converter_class()
        opts = {
            "imgsz": args.imgsz,
            "opset": args.opset,
            "dynamic": args.dynamic,
        }
        if args.model_type:
            opts["model_type"] = args.model_type
        result_meta = converter.convert(
            pt_path=args.input,
            onnx_path=args.output,
            **opts,
        )
    except Exception as exc:
        logging.getLogger("pt2onnx").error("Conversion failed: %s", exc, exc_info=True)
        sys.exit(1)

    nc = result_meta.get("nc", "?")
    imgsz = result_meta.get("imgsz", "?")
    opset = result_meta.get("opset", "?")
    variant = result_meta.get("variant", "?")
    if variant != "?":
        print(
            f"✓ Export succeeded via {args.converter}: variant={variant}, opset={opset}\n"
            f"  Output prefix: {args.output}\n"
            f"  Metadata: {args.output}.meta.json"
        )
    else:
        print(
            f"✓ Export succeeded via {args.converter}: nc={nc}, imgsz={imgsz}, opset={opset}\n"
            f"  Output:  {args.output}\n"
            f"  Metadata: {args.output}.meta.json"
        )


if __name__ == "__main__":
    main()
