import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestInjectMSCA:
    def test_inject_msca_missing_file(self):
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        with patch("yolo_converter.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="MSCA.py"):
                converter._inject_msca()

    def test_injects_msca_successfully(self):
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader
        mock_loader.name = "MSCA"

        with (
            patch("yolo_converter.Path.exists", return_value=True),
            patch("yolo_converter.Path.suffix", create=True, new_callable=lambda: ".py"),
            patch("importlib.util.spec_from_file_location", return_value=mock_spec),
            patch("importlib.util.module_from_spec") as mock_mod_from_spec,
        ):
            mock_mod = MagicMock()
            mock_mod.MSCAAttention = MagicMock()
            mock_mod_from_spec.return_value = mock_mod

            from yolo_converter import YoloConverter

            converter = YoloConverter()
            converter._inject_msca()


class TestConvert:
    @pytest.fixture(autouse=True)
    def _patch_path_exists(self):
        """Make Path.exists return True so pt-existence check passes."""
        with patch("yolo_converter.Path.exists", return_value=True):
            yield

    def test_pt_not_found(self):
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        with patch("yolo_converter.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="checkpoint"):
                converter.convert(pt_path="nonexistent.pt", onnx_path="out.onnx")

    def test_nc_derived_from_output0(self, mock_msca_injection, mock_yolo, mock_onnx_session):
        mock_onnx_session(inp_shape=[1, 3, 640, 640], out0_shape=[1, 37, 8400])
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        meta = converter.convert(pt_path="m.pt", onnx_path="m.onnx")
        assert meta["nc"] == 1

    def test_nc_twenty_classes(self, mock_msca_injection, mock_yolo, mock_onnx_session):
        mock_onnx_session(inp_shape=[1, 3, 640, 640], out0_shape=[1, 56, 8400])
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        meta = converter.convert(pt_path="m.pt", onnx_path="m.onnx")
        assert meta["nc"] == 20

    def test_static_hw(self, mock_msca_injection, mock_yolo, mock_onnx_session):
        mock_onnx_session(inp_shape=[1, 3, 640, 640], out0_shape=[1, 37, 8400])
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        meta = converter.convert(pt_path="m.pt", onnx_path="m.onnx")
        assert meta["model_h"] == 640
        assert meta["model_w"] == 640

    def test_dynamic_hw_fallback_to_imgsz(self, mock_msca_injection, mock_yolo, mock_onnx_session):
        mock_onnx_session(inp_shape=["B", 3, "H", "W"], out0_shape=[1, 37, 8400])
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        meta = converter.convert(pt_path="m.pt", onnx_path="m.onnx", imgsz=1024)
        assert meta["model_h"] == 1024
        assert meta["model_w"] == 1024

    def test_symbolic_feature_dim_raises(self, mock_msca_injection, mock_yolo, mock_onnx_session):
        mock_onnx_session(inp_shape=[1, 3, 640, 640], out0_shape=["B", "C", 8400])
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        with pytest.raises(RuntimeError, match="symbolic"):
            converter.convert(pt_path="m.pt", onnx_path="m.onnx")

    def test_parameters_passed_to_ultralytics(self, mock_msca_injection, mock_onnx_session):
        mock_onnx_session(inp_shape=[1, 3, 640, 640], out0_shape=[1, 37, 8400])
        from yolo_converter import YoloConverter

        with patch.dict("sys.modules"):
            mock_ultralytics = MagicMock()
            mock_instance = MagicMock()
            mock_ultralytics.YOLO.return_value = mock_instance
            sys.modules["ultralytics"] = mock_ultralytics

            converter = YoloConverter()
            converter.convert(pt_path="m.pt", onnx_path="m.onnx", imgsz=640, opset=15, dynamic=False)
            mock_instance.export.assert_called_once_with(
                format="onnx", imgsz=640, opset=15, dynamic=False, simplify=True
            )

    def test_onnx_file_moved(self, mock_msca_injection, mock_yolo, mock_onnx_session):
        mock_onnx_session(inp_shape=[1, 3, 640, 640], out0_shape=[1, 37, 8400])
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        with patch("yolo_converter.Path.rename"):
            meta = converter.convert(pt_path="m.pt", onnx_path="out/m.onnx")
        assert meta["source_pt"] == str(Path("m.pt").resolve())

    def test_metadata_json_content(self, mock_msca_injection, mock_yolo, mock_onnx_session):
        actual_written = {}

        def fake_write_text(content):
            actual_written["data"] = json.loads(content)

        mock_onnx_session(inp_shape=[1, 3, 640, 640], out0_shape=[1, 56, 8400])
        from yolo_converter import YoloConverter

        with patch("pathlib.Path.write_text", side_effect=fake_write_text):
            converter = YoloConverter()
            converter.convert(pt_path="/abs/m.pt", onnx_path="/abs/m.onnx", imgsz=1024, opset=15, dynamic=True)

        meta = actual_written["data"]
        assert meta["nc"] == 20
        assert meta["model_h"] == 640
        assert meta["model_w"] == 640
        assert meta["imgsz"] == 1024
        assert meta["opset"] == 15
        assert meta["dynamic"] is True
        assert "input_name" in meta
        assert "output_names" in meta
        assert "source_pt" in meta

    def test_dynamic_flag_in_meta(self, mock_msca_injection, mock_yolo, mock_onnx_session):
        mock_onnx_session(inp_shape=[1, 3, 640, 640], out0_shape=[1, 37, 8400])
        from yolo_converter import YoloConverter

        converter = YoloConverter()
        meta = converter.convert(pt_path="m.pt", onnx_path="m.onnx", dynamic=False)
        assert meta["dynamic"] is False
