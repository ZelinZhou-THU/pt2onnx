# pt2onnx-converter

Universal `.pt` → ONNX model conversion tool with companion `.meta.json` sidecars for zero-configuration downstream inference.

## Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| YOLO-seg (v8 / 11 / …) | ✅ Supported | Any segmentation model trained with ultralytics |
| YOLO-seg + MSCA | ✅ Supported | Custom SegNeXt attention module injected automatically |
| RF-DETR | 🔜 Planned | — |
| SAM | 🔜 Planned | — |

ultralytics auto-detects the model architecture from the `.pt` checkpoint, so any YOLO variant it supports will work out of the box.

## Features

- **MSCA module injection** — automatically registers the SegNeXt MSCAAttention custom module before export, so checkpoints that reference custom attention layers export cleanly. If the checkpoint does not use MSCA, the injection is harmless.
- **Dynamic batch** — exports with a symbolic batch axis by default, enabling true batched inference on the ONNX runtime side.
- **Meta sidecar** — writes `<model>.onnx.meta.json` containing `nc`, `model_h`, `model_w`, `opset`, `dynamic`, and I/O tensor names.

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- ultralytics == 8.3.162
- onnxruntime >= 1.17.0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python cli.py --input models/best.pt --output models/best.onnx
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input `.pt` checkpoint path | Required |
| `--output` | Output `.onnx` path | Required |
| `--imgsz` | Model input resolution (square) | `640` |
| `--opset` | ONNX opset version | `17` |
| `--dynamic` | Export with symbolic batch axis | `True` |
| `--no-dynamic` | Fix all dimensions (batch=1) | — |

### Output

The converter produces two files:

- `<name>.onnx` — the ONNX model graph
- `<name>.onnx.meta.json` — metadata sidecar:

```json
{
  "nc": 1,
  "model_h": 640,
  "model_w": 640,
  "imgsz": 640,
  "opset": 17,
  "dynamic": true,
  "input_name": "images",
  "output_names": ["output0", "output1"],
  "source_pt": "/path/to/best.pt"
}
```

The `nc` (number of classes) is derived from the ONNX output tensor shape. For segmentation models the formula is `nc = output0_dim[1] - 36` (4 bbox coefficients + 32 mask coefficients). If you add a converter for a non-segmentation architecture, this calculation must be adjusted accordingly.

## Adding New Converters

To support a new model framework (e.g. RF-DETR, SAM):

### 1. Create a converter module

Add `<framework>_converter.py` alongside `yolo_converter.py`. A converter must expose a `convert()` method that:

```python
class MyConverter:
    def convert(self, pt_path: str, onnx_path: str, imgsz: int,
                opset: int, dynamic: bool) -> dict:
        """
        Convert a .pt checkpoint to .onnx and return a meta dict.

        Required meta fields:
            nc        – number of classes (int)
            model_h   – input height (int)
            model_w   – input width (int)
            imgsz     – input resolution as passed (int)
            opset     – ONNX opset version (int)
            dynamic   – whether batch axis is symbolic (bool)
            input_name  – name of the input tensor (str)
            output_names – list of output tensor names (list[str])

        The meta dict is written to <onnx_path>.meta.json by the caller
        if the converter does not write it itself.
        """
```

### 2. Handle custom modules (if needed)

If the model uses custom layers not known to the framework at deserialization time, inject them before loading — see `_inject_msca()` in `yolo_converter.py` for an example of `sys.modules` patching.

### 3. Register in CLI

Add a `--framework` argument to `cli.py` and dispatch to the appropriate converter:

```python
if args.framework == "yolo":
    from yolo_converter import YoloConverter
    converter = YoloConverter()
elif args.framework == "rfdetr":
    from rfdetr_converter import RfdetrConverter
    converter = RfdetrConverter()
```

### 4. Add dependencies

List framework-specific packages in `requirements.txt` with license annotations. Keep in mind that any dependency with a copyleft license (AGPL, GPL) will apply to this entire tool — that is acceptable here since pt2onnx-converter is already AGPL-3.0.

## File Overview

| File | Description |
|------|-------------|
| `cli.py` | Command-line entry point |
| `yolo_converter.py` | YOLO-seg conversion logic (ultralytics export + meta sidecar) |
| `MSCA.py` | MSCAAttention module (from SegNeXt, NeurIPS 2022) |
| `requirements.txt` | Python dependencies |

## License

This project is licensed under the **GNU Affero General Public License v3.0** ([LICENSE](LICENSE)).

This dependency arises from [ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0.

### Third-Party Notices

This project includes code derived from:

- **SegNeXt** — *Rethinking Convolutional Attention Design for Semantic Segmentation* (NeurIPS 2022)
  - Copyright 2022 SegNeXt Authors (Meng-Hao Guo et al.)
  - Licensed under the Apache License, Version 2.0
  - Source: https://github.com/Visual-Attention-Network/SegNeXt
  - See [NOTICE](NOTICE) for full license text and change details.

### Dependency Licenses

| Package | License |
|---------|---------|
| ultralytics | AGPL-3.0 |
| PyTorch | Apache-2.0 (BSD-3-Clause components) |
| onnxruntime | MIT |
