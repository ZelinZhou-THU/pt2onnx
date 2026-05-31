import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _setup_sys_path():
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    yield
    if root in sys.path:
        sys.path.remove(root)


@pytest.fixture
def mock_msca_injection():
    with patch("yolo_converter.YoloConverter._inject_msca") as m:
        yield m


@pytest.fixture
def mock_yolo():
    """Mock ultralytics module (imported inside convert() body)."""
    mock_ultralytics = MagicMock()
    mock_yolo_instance = MagicMock()
    mock_ultralytics.YOLO.return_value = mock_yolo_instance
    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
        yield mock_yolo_instance


@pytest.fixture
def mock_onnx_session():
    """Mock onnxruntime module (imported inside convert() body).

    Returns a callable ``set_shapes(inp_shape, out0_shape)`` that configures
    the mock InferenceSession to return those I/O shapes.
    """
    mock_ort = MagicMock()

    def _set_shapes(inp_shape, out0_shape):
        inp = MagicMock()
        inp.name = "images"
        inp.shape = inp_shape

        out0 = MagicMock()
        out0.name = "output0"
        out0.shape = out0_shape

        out1 = MagicMock()
        out1.name = "output1"
        out1.shape = [1, 32, 160, 160]

        sess = MagicMock()
        sess.get_inputs.return_value = [inp]
        sess.get_outputs.return_value = [out0, out1]
        mock_ort.InferenceSession.return_value = sess
        return sess

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        yield _set_shapes
