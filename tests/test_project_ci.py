
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required = [
        "webapp/app.py",
        "webapp/inference.py",
        "webapp/database.py",
        "Dockerfile",
        "requirements-docker.txt",
        "models.dvc",
    ]

    for file in required:
        assert (ROOT / file).is_file(), f"Missing: {file}"


def test_webapp_python_syntax():
    for file in (ROOT / "webapp").glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"))


def test_dockerfile_contains_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "HEALTHCHECK" in dockerfile


def test_dockerfile_uses_non_root_user():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in dockerfile
