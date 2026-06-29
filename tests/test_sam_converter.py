import json
import sys
import types
from pathlib import Path


def _install_fakes(monkeypatch, tmp_path):
    """Inject fake torch and fake segment_anything into sys.modules, return recorder."""
    calls = {"export": []}

    fake_torch = types.ModuleType("torch")

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    fake_torch.no_grad = lambda: _Ctx()
    fake_torch.randn = lambda *shape, **kw: ("randn", shape)
    fake_torch.tensor = lambda x, dtype=None: ("tensor", x)
    fake_torch.randint = lambda low, high, size, **kw: ("randint", size)
    fake_torch.float = "float32"

    def _export(model, dummy, path, **kw):
        Path(path).write_bytes(b"FAKE_ONNX")
        calls["export"].append({"path": path, **kw})
    fake_torch.onnx = types.SimpleNamespace(export=_export)

    fake_sa = types.ModuleType("segment_anything")
    fake_utils = types.ModuleType("segment_anything.utils")
    fake_onnx_mod = types.ModuleType("segment_anything.utils.onnx")

    class _PromptEnc:
        embed_dim = 256
        image_embedding_size = (64, 64)
    class _Sam:
        def __init__(self): self.image_encoder = object(); self.prompt_encoder = _PromptEnc()
        def to(self, x): return self
        def eval(self): return self
    fake_sa.sam_model_registry = {"vit_h": lambda checkpoint: _Sam(),
                                  "vit_l": lambda checkpoint: _Sam(),
                                  "vit_b": lambda checkpoint: _Sam()}

    class _SamOnnxModel:
        def __init__(self, model, return_single_mask=False): self.model = model
        def __call__(self, **kw): return None
    fake_onnx_mod.SamOnnxModel = _SamOnnxModel
    fake_utils.onnx = fake_onnx_mod
    fake_sa.utils = fake_utils

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "segment_anything", fake_sa)
    monkeypatch.setitem(sys.modules, "segment_anything.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "segment_anything.utils.onnx", fake_onnx_mod)
    return calls


def test_sam_converter_writes_encoder_decoder_and_meta(monkeypatch, tmp_path):
    calls = _install_fakes(monkeypatch, tmp_path)
    from sam_converter import SamConverter

    (tmp_path / "ckpt.pth").write_bytes(b"FAKE_CHECKPOINT")
    prefix = str(tmp_path / "sam_vit_h")
    meta = SamConverter().convert(
        pt_path=str(tmp_path / "ckpt.pth"),
        onnx_path=prefix,
        model_type="vit_h",
        opset=17,
    )

    exported = {Path(c["path"]).name for c in calls["export"]}
    assert "sam_vit_h_encoder.onnx" in exported
    assert "sam_vit_h_decoder.onnx" in exported
    assert (tmp_path / "sam_vit_h_encoder.onnx").read_bytes() == b"FAKE_ONNX"
    assert (tmp_path / "sam_vit_h_decoder.onnx").exists()
    meta_file = tmp_path / "sam_vit_h.meta.json"
    assert meta_file.exists()
    parsed = json.loads(meta_file.read_text())
    assert parsed["variant"] == "vit_h"
    assert parsed["encoder_path"].endswith("sam_vit_h_encoder.onnx")
    assert parsed["decoder_path"].endswith("sam_vit_h_decoder.onnx")
    assert parsed["decoder_inputs"] == ["image_embeddings", "point_coords", "point_labels",
                                        "mask_input", "has_mask_input", "orig_im_size"]
    assert parsed["decoder_outputs"] == ["masks", "iou_predictions", "low_res_masks"]
    assert meta["variant"] == "vit_h"


def test_sam_converter_strips_onnx_suffix_from_prefix(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path)
    from sam_converter import SamConverter
    (tmp_path / "x.pth").write_bytes(b"FAKE")
    SamConverter().convert(pt_path=str(tmp_path / "x.pth"), onnx_path=str(tmp_path / "out.onnx"),
                           model_type="vit_b", opset=17)
    assert (tmp_path / "out_encoder.onnx").exists()
    assert (tmp_path / "out_decoder.onnx").exists()
    assert (tmp_path / "out.meta.json").exists()
