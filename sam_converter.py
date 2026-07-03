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

"""SAM (.pth) -> encoder + decoder ONNX converter.

官方 segment-anything 仅提供 decoder 导出脚本；encoder 由本插件自导。
两个 ONNX + 一个 .meta.json。依赖 torch + segment_anything（Apache-2.0）。
仅子进程调用，勿 import 进主应用。
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("pt2onnx")

CONVERTER_META = {
    "name": "sam",
    "display_name": "SAM (Segment Anything)",
    "source_formats": [".pth"],
    "target_format": ".onnx",
    "dependencies": ["segment_anything", "torch"],
    "description": "Segment Anything ViT-B/L/H (.pth -> encoder+decoder ONNX)",
    "default_imgsz": 1024,
    "supports_dynamic": False,
    "class_name": "SamConverter",
    "multi_output": True,
}


class SamConverter:
    """Convert SAM .pth checkpoint to encoder + decoder ONNX files."""

    def convert(self, pt_path: str, onnx_path: str,
                model_type: str = "vit_h", opset: int = 17, **opts) -> dict:
        import torch
        from segment_anything import sam_model_registry
        from segment_anything.utils.onnx import SamOnnxModel

        pt = Path(pt_path).resolve()
        if not pt.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {pt}")

        prefix = onnx_path[:-5] if onnx_path.endswith(".onnx") else onnx_path
        encoder_out = f"{prefix}_encoder.onnx"
        decoder_out = f"{prefix}_decoder.onnx"
        Path(encoder_out).parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading SAM checkpoint: %s (type=%s)", pt_path, model_type)
        sam = sam_model_registry[model_type](checkpoint=pt_path)
        # CPU export keeps ONNX weights identical across host machines and
        # avoids CUDA version pin mismatch. Conversion time is acceptable:
        # ~90s for ViT-H.
        sam.to("cpu").eval()

        # ---- encoder ----
        enc_dummy = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            # dynamo=False: SAM decoder uses data-dependent shape ops
            # (e.g. `int(prepadded_size[0])` in mask_postprocessing) that the
            # new torch.export-based path cannot trace. Fall back to the
            # classic TorchScript exporter which handles them correctly.
            torch.onnx.export(
                sam.image_encoder, enc_dummy, encoder_out,
                input_names=["images"], output_names=["image_embeddings"],
                opset_version=opset, do_constant_folding=True,
                dynamo=False,
            )
        logger.info("Encoder exported -> %s", encoder_out)

        # ---- decoder (return_single_mask=False -> 3 candidates) ----
        onnx_model = SamOnnxModel(model=sam, return_single_mask=False)
        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        mask_in_size = [4 * x for x in embed_size]
        dummy_inputs = {
            "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords": torch.randint(0, 1024, (1, 5, 2), dtype=torch.float),
            "point_labels": torch.randint(0, 4, (1, 5), dtype=torch.float),
            "mask_input": torch.randn(1, 1, *mask_in_size, dtype=torch.float),
            "has_mask_input": torch.tensor([1], dtype=torch.float),
            "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
        }
        with torch.no_grad():
            _ = onnx_model(**dummy_inputs)  # trace warmup
            # dynamo=False: see comment above on encoder export.
            torch.onnx.export(
                onnx_model, tuple(dummy_inputs.values()), decoder_out,
                input_names=list(dummy_inputs.keys()),
                output_names=["masks", "iou_predictions", "low_res_masks"],
                opset_version=opset, do_constant_folding=True,
                dynamic_axes={"point_coords": {1: "num_points"},
                              "point_labels": {1: "num_points"}},
                dynamo=False,
            )
        logger.info("Decoder exported -> %s", decoder_out)

        meta = {
            "schema": "sam",  # explicit schema tag for type detection
            "variant": model_type,
            "encoder_path": encoder_out,
            "decoder_path": decoder_out,
            "encoder_input": "images",
            "encoder_output": "image_embeddings",
            "decoder_inputs": list(dummy_inputs.keys()),
            "decoder_outputs": ["masks", "iou_predictions", "low_res_masks"],
            "opset": opset,
            "source_pt": str(pt_path),
        }
        Path(f"{prefix}.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))
        return meta
