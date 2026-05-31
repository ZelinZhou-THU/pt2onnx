"""Smoke tests that verify imports work correctly."""


def test_import_yolo_converter():
    from yolo_converter import YoloConverter

    assert YoloConverter is not None


def test_import_cli():
    import cli

    assert hasattr(cli, "main")


def test_requirements_txt_exists():
    from pathlib import Path

    req = Path(__file__).resolve().parent.parent / "requirements.txt"
    assert req.exists()
    content = req.read_text(encoding="utf-8")
    assert "ultralytics" in content
    assert "torch" in content
    assert "onnxruntime" in content
