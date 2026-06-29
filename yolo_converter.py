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
YOLO-seg .pt → .onnx converter with MSCA custom module injection.

Supports all YOLO segmentation models trained with ultralytics (YOLOv8-seg,
YOLO11-seg, etc.). Custom modules (e.g. MSCA) are injected into ultralytics
at load time so that checkpoints referencing them can be exported cleanly.

Depends on ultralytics (AGPL-3.0). Call via subprocess only; do not import
into projects with incompatible licenses.
"""

import json
import logging
import sys
import types
from pathlib import Path

logger = logging.getLogger("pt2onnx")

# Plugin metadata contract: discovered by ConverterRegistry / cli --list-converters.
# Keep keys stable; this is the public contract for adding new converter modules
# (UNet, MMDet, etc.). See NanoAnalyst/.../converter_registry.py.
#
# MUST be a literal dict (not dict() constructor or variable concatenation),
# because cli._discover_converters() parses it via ast.literal_eval().
CONVERTER_META = {
    "name": "yolo_seg",
    "display_name": "YOLO-seg (v5-v11)",
    "source_formats": [".pt"],
    "target_format": ".onnx",
    "dependencies": ["ultralytics", "torch"],
    "description": "YOLO instance segmentation (v5-v11) (.pt \u2192 .onnx)",
    "default_imgsz": 640,
    "supports_dynamic": True,
    "class_name": "YoloConverter",
}


class YoloConverter:
    """Convert a YOLO-seg .pt checkpoint to ONNX."""

    def _inject_msca(self) -> None:
        """Load the bundled MSCA.py and inject MSCAAttention into ultralytics modules."""
        local_msca = Path(__file__).parent / "MSCA.py"
        if not local_msca.exists():
            raise FileNotFoundError(f"MSCA.py not found at {local_msca}")

        spec_dir = str(Path(__file__).parent)
        if spec_dir not in sys.path:
            sys.path.insert(0, spec_dir)
        import importlib.util
        spec = importlib.util.spec_from_file_location("MSCA", local_msca)
        msca_src = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(msca_src)

        MSCAAttention = msca_src.MSCAAttention

        import ultralytics.nn as unn

        attention_pkg = types.ModuleType("ultralytics.nn.attention")
        attention_pkg.__path__ = []
        attention_pkg.__package__ = "ultralytics.nn"

        msca_mod = types.ModuleType("ultralytics.nn.attention.MSCA")
        msca_mod.MSCAAttention = MSCAAttention
        msca_mod.__package__ = "ultralytics.nn.attention"

        attention_pkg.MSCA = msca_mod
        unn.attention = attention_pkg
        sys.modules["ultralytics.nn.attention"] = attention_pkg
        sys.modules["ultralytics.nn.attention.MSCA"] = msca_mod

        logger.info("MSCA injected successfully")

    def convert(
        self,
        pt_path: str,
        onnx_path: str,
        imgsz: int = 640,
        opset: int = 18,
        dynamic: bool = True,
        **opts,
    ) -> dict:
        """Convert best.pt → best.onnx.

        dynamic=True (default): symbolic batch axis for true batched inference
            via OnnxBackend.predict_batch.
        dynamic=False: all dimensions fixed (batch=1) for runtimes without
            dynamic axis support.

        Writes <onnx_path>.meta.json with exact nc / model_h / model_w.
        """
        pt = Path(pt_path).resolve()
        out = Path(onnx_path).resolve()

        if not pt.exists():
            raise FileNotFoundError(f".pt checkpoint not found: {pt}")

        self._inject_msca()

        from ultralytics import YOLO

        logger.info("Loading .pt model: %s", pt)
        model = YOLO(str(pt))

        logger.info("Exporting ONNX (imgsz=%d, opset=%d, dynamic=%s) ...", imgsz, opset, dynamic)
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            dynamic=dynamic,
            simplify=True,
        )

        # ultralytics writes the .onnx next to the .pt with the same stem
        default_out = pt.with_suffix(".onnx")
        if default_out != out:
            out.parent.mkdir(parents=True, exist_ok=True)
            default_out.rename(out)
            logger.info("Moved to target path: %s", out)

        # Verify export and extract exact nc / model dimensions
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        inp_shape = sess.get_inputs()[0].shape
        out0_shape = sess.get_outputs()[0].shape

        # Feature dim (index 1) must be a fixed int regardless of dynamic setting.
        if not isinstance(out0_shape[1], int):
            raise RuntimeError(
                f"output0 feature dim is still symbolic ({out0_shape[1]}); nc cannot be determined."
            )
        nc = int(out0_shape[1]) - 36   # 4 bbox + 32 mask coefficients

        # H/W may be symbolic when dynamic=True; fall back to imgsz.
        model_h = int(inp_shape[2]) if isinstance(inp_shape[2], int) else imgsz
        model_w = int(inp_shape[3]) if isinstance(inp_shape[3], int) else imgsz

        meta = {
            "nc": nc,
            "model_h": model_h,
            "model_w": model_w,
            "imgsz": imgsz,
            "opset": opset,
            "dynamic": dynamic,
            "input_name": sess.get_inputs()[0].name,
            "output_names": [o.name for o in sess.get_outputs()],
            "source_pt": str(pt),
        }
        meta_path = Path(str(out) + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        logger.info(
            "Export complete: %s | nc=%d | dynamic=%s | meta → %s",
            out.name, nc, dynamic, meta_path.name,
        )
        return meta
