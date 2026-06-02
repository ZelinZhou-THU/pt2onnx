import sys
from unittest.mock import MagicMock, patch

import pytest


def _mock_resolver(return_value=None):
    """Return a patch context for cli._resolve_converter_class.

    The new cli.py dispatches through _resolve_converter_class() via
    importlib rather than doing a direct import, so patching the module-level
    class (yolo_converter.YoloConverter) no longer intercepts the call.
    Patching _resolve_converter_class is the correct seam.
    """
    mock_cls = MagicMock()
    mock_cls.return_value.convert.return_value = return_value or {"nc": 1, "imgsz": 640, "opset": 17}
    return patch("cli._resolve_converter_class", return_value=mock_cls), mock_cls


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

        p, mock_cls = _mock_resolver()
        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            p,
        ):
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["imgsz"] == 640

    def test_imgsz_custom(self):
        import cli

        p, mock_cls = _mock_resolver({"nc": 1, "imgsz": 1024, "opset": 17})
        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx", "--imgsz", "1024"]),
            p,
        ):
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["imgsz"] == 1024

    def test_opset_default(self):
        import cli

        p, mock_cls = _mock_resolver()
        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            p,
        ):
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["opset"] == 18

    def test_dynamic_default_true(self):
        import cli

        p, mock_cls = _mock_resolver()
        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            p,
        ):
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["dynamic"] is True

    def test_no_dynamic(self):
        import cli

        p, mock_cls = _mock_resolver()
        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx", "--no-dynamic"]),
            p,
        ):
            cli.main()
            kwargs = mock_cls.return_value.convert.call_args[1]
            assert kwargs["dynamic"] is False

    def test_main_success_output(self, capsys):
        import cli

        p, mock_cls = _mock_resolver()
        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            p,
        ):
            cli.main()
        captured = capsys.readouterr()
        assert "Export succeeded" in captured.out

    def test_main_handles_exception(self):
        import cli

        p, mock_cls = _mock_resolver()
        mock_cls.return_value.convert.side_effect = RuntimeError("boom")
        with (
            patch.object(sys, "argv", ["cli.py", "--input", "m.pt", "--output", "m.onnx"]),
            p,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli.main()
        assert exc_info.value.code == 1
