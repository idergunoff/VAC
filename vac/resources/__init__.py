"""Installed application resources, independent of the working directory."""
from pathlib import Path

def template_path():
    return Path(__file__).resolve().parent / "template.docx"
