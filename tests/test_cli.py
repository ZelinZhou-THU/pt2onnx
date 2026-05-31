import sys
from unittest.mock import patch

import pytest


class TestCLIArguments:
    def test_input_required(self):
        import cli

        with patch.object(sys, "argv", ["cli.py"]), pytest.raises(SystemExit):
            cli.main()

    def test_output_required(self):
        import cli

        with (
            patch.object(sys, "argv", ["cli.py", "--input", "model.pt"]),
            pytest.raises(SystemExit),
        ):
            cli.main()

    def test_imgsz_default(self):
        import cli

        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            patch("yolo_converter.YoloConverter") as mock_cls,
        ):
            mock_cls.return_value.convert.return_value = {"nc": 1, "imgsz": 640, "opset": 17}
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["imgsz"] == 640

    def test_imgsz_custom(self):
        import cli

        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx", "--imgsz", "1024"]),
            patch("yolo_converter.YoloConverter") as mock_cls,
        ):
            mock_cls.return_value.convert.return_value = {"nc": 1, "imgsz": 1024, "opset": 17}
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["imgsz"] == 1024

    def test_opset_default(self):
        import cli

        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            patch("yolo_converter.YoloConverter") as mock_cls,
        ):
            mock_cls.return_value.convert.return_value = {"nc": 1, "imgsz": 640, "opset": 17}
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["opset"] == 17

    def test_dynamic_default_true(self):
        import cli

        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            patch("yolo_converter.YoloConverter") as mock_cls,
        ):
            mock_cls.return_value.convert.return_value = {"nc": 1, "imgsz": 640, "opset": 17}
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["dynamic"] is True

    def test_no_dynamic(self):
        import cli

        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx", "--no-dynamic"]),
            patch("yolo_converter.YoloConverter") as mock_cls,
        ):
            mock_cls.return_value.convert.return_value = {"nc": 1, "imgsz": 640, "opset": 17}
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["dynamic"] is False

    def test_main_success_output(self, capsys):
        import cli

        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            patch("yolo_converter.YoloConverter") as mock_cls,
        ):
            mock_cls.return_value.convert.return_value = {"nc": 1, "imgsz": 640, "opset": 17}
            cli.main()
        captured = capsys.readouterr()
        assert "Export succeeded" in captured.out

    def test_main_handles_exception(self):
        import cli

        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            patch("yolo_converter.YoloConverter") as mock_cls,
        ):
            mock_cls.return_value.convert.side_effect = RuntimeError("boom")
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 1
