"""Extract the <script> blocks from ui.html / obs.html and syntax-check them with Node."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
failed = False
for name in ("ui.html", "obs.html"):
    html = (root / name).read_text(encoding="utf-8")
    assert "<nhtml" not in html, f"{name}: broken <html> tag"
    for i, code in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S)):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
            fh.write(code)
        result = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True)
        print(f"{name} script {i}: {'ok' if result.returncode == 0 else 'FAILED'}")
        if result.returncode:
            print(result.stderr)
            failed = True
sys.exit(1 if failed else 0)
