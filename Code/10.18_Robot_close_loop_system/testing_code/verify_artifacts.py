"""Check saved runs, source provenance, Notebook outputs, and local links."""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

import nbformat

from experiment import ROOT, save_json


def main():
    result = ROOT / "10.18_result"
    checks = {}
    for name in ["debug", "main", "repeat", "seed_19", "seed_20"]:
        folder = result / name
        manifest = json.loads((folder / "manifest.json").read_text())
        checks[f"{name}_files"] = all(hashlib.sha256((folder / path).read_bytes()).hexdigest() == digest
                                     for path, digest in manifest.items())
        checks[f"{name}_acceptance"] = json.loads((folder / "verification.json").read_text())["passed"]
    notebook = nbformat.read(ROOT / "10.18_test_report.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    outputs = [o for cell in code for o in cell.outputs]
    checks["notebook_executed"] = all(c.execution_count is not None for c in code)
    checks["notebook_no_errors"] = not any(o.output_type == "error" for o in outputs)
    checks["six_embedded_figures"] = sum("image/png" in o.get("data", {}) for o in outputs) == 6
    text = (ROOT / "README.md").read_text() + "\n" + "\n".join(
        c.source for c in notebook.cells if c.cell_type == "markdown")
    links = re.findall(r"\]\(([^)]+)\)", text)
    checks["local_links_exist"] = all((ROOT / unquote(link)).exists() for link in links
                                      if "://" not in link and not link.startswith("#"))
    hash_path = result / "verification/source_hashes.json"
    hashes = json.loads(hash_path.read_text())
    checks["executed_source_unchanged"] = all(
        hashlib.sha256((ROOT.parent / path).read_bytes()).hexdigest() == digest
        for path, digest in hashes.items())
    # Include the report tools added after the numerical runs. Existing hashes
    # must match before recording those extra source files.
    if checks["executed_source_unchanged"]:
        for path in sorted((ROOT / "testing_code").glob("*")):
            if path.suffix in {".py", ".json", ".txt"}:
                hashes[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        save_json(hash_path, hashes)
    save_json(result / "verification/artifacts.json", {"passed": all(checks.values()),
                                                       "checks": checks,
                                                       "code_cells": len(code),
                                                       "local_links": len(links)})
    print(json.dumps(checks, indent=2))
    assert all(checks.values()), "Artifact verification failed"


if __name__ == "__main__":
    main()
