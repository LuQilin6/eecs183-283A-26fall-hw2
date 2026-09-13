"""Check HW2 files and build a submission ZIP; never execute submitted code."""
import argparse
import ast
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


REQUIRED_RESULTS = {
    "bpe_merges.json": dict,
    "bpe_vocab.json": dict,
    "en_tokenizer.json": dict,
    "neighbor_context.json": list,
    "solve_analogy.json": list,
    "skipgram_scores.npy": (65536, 11),
    "W_en_fr.npy": (300, 300),
    "W_en_ja.npy": (300, 300),
}


def build_submission(notebook="hw2.ipynb", report="report.pdf", results="results",
                     out="submission.zip"):
    """Package available work, warning about incomplete or invalid answers."""
    notebook, results, out = map(Path, (notebook, results, out))
    report = Path(report) if report is not None else None
    errors = []
    warnings = []
    try:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        if (not isinstance(document, dict) or document.get("nbformat") != 4
                or not isinstance(document.get("cells"), list)):
            raise ValueError("expected a Jupyter notebook in version 4 format")
        for index, cell in enumerate(document["cells"], 1):
            if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown", "raw"}:
                raise ValueError(f"invalid notebook cell {index}")
            source = cell.get("source")
            if not (isinstance(source, str) or isinstance(source, list)
                    and all(isinstance(line, str) for line in source)):
                raise ValueError(f"invalid source in notebook cell {index}")
            if cell["cell_type"] == "code":
                try:
                    ast.parse("".join(source))
                except SyntaxError as error:
                    warnings.append(f"Notebook cell {index}: {error.msg}; this cell cannot be graded.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{notebook}: {error}")
    try:
        if report is None:
            raise ValueError("report.pdf omitted; written answers will not be available for grading")
        with report.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("expected a PDF file")
    except (OSError, ValueError) as error:
        warnings.append(f"{report or 'report.pdf'}: {error}")

    for name, expected in REQUIRED_RESULTS.items():
        path = results / name
        try:
            if path.resolve().parent != results.resolve():
                errors.append(f"{path}: result file points outside the results directory")
                continue
            if name.endswith(".npy"):
                array = np.load(path, allow_pickle=False)
                if array.shape != expected:
                    raise ValueError(f"expected shape {expected}, found {array.shape}")
                if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                    raise ValueError("expected finite, real numeric values")
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, expected) or not value:
                    raise ValueError(f"expected a nonempty {expected.__name__}")
                if name == "solve_analogy.json" and (
                    len(value) != 6 or not all(isinstance(x, str) and x.strip() for x in value)
                ):
                    raise ValueError("expected six nonempty answer strings")
                if name == "neighbor_context.json" and not all(
                    isinstance(pair, list) and len(pair) == 2
                    and all(type(x) is int and x >= 0 for x in pair) for pair in value
                ):
                    raise ValueError("expected pairs of nonnegative integer token IDs")
        except (OSError, ValueError, TypeError, EOFError) as error:
            warnings.append(f"{path}: {error}")

    if out.suffix.lower() != ".zip":
        errors.append("The output filename must end in .zip.")
    entries = [(notebook, "hw2.ipynb")]
    if report is not None and report.is_file():
        entries.append((report, "report.pdf"))
    for path in sorted(results.rglob("*")):
        relative = path.relative_to(results)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_file() and path.resolve() != out.resolve():
            if not path.resolve().is_relative_to(results.resolve()):
                errors.append(f"{path}: file points outside the results directory")
            else:
                entries.append((path, f"results/{relative.as_posix()}"))
    if errors:
        raise ValueError("Submission ZIP was not created or updated:\n" +
                         "\n".join(f"  - {error}" for error in errors))

    if warnings:
        print("Packaging partial work. Missing or invalid answers may lose the corresponding credit:")
        for warning in warnings:
            print(f"  - {warning}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Replace an older ZIP only after the new archive is fully written.
    handle, temporary = tempfile.mkstemp(suffix=".zip", dir=out.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, name in entries:
                archive.write(path, name)
        os.replace(temporary, out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Created {out} ({len(entries)} files). Format checks do not grade correctness.")
    return out


def from_colab(include_report=True):
    """Capture available work; optionally request a report, then download a ZIP."""
    from google.colab import _message, files

    # A clone contains the starter notebook, not the student's browser edits.
    # This request must run in the notebook kernel, not a subprocess.
    try:
        response = _message.blocking_request("get_ipynb", timeout_sec=30)
        document = response["ipynb"]
        if document.get("nbformat") != 4 or not document.get("cells"):
            raise ValueError("invalid notebook response")
    except Exception as error:
        raise RuntimeError(
            "Could not read the open notebook. Download it using File > Download > "
            "Download .ipynb, then follow the README's manual packaging instructions."
        ) from error

    saved = Path("/content/drive/MyDrive/cs183-hw2")
    if not saved.is_dir():
        raise RuntimeError("Run the setup cell first to connect Google Drive.")
    notebook = saved / "hw2.ipynb"
    notebook.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    report = None
    if include_report:
        print("Select your report PDF, or cancel the upload to submit without a report.")
        uploaded = files.upload()
        if uploaded:
            if len(uploaded) != 1 or not next(iter(uploaded)).lower().endswith(".pdf"):
                raise ValueError("Select one PDF, or set include_report=False and rerun the cell.")
            report = saved / "report.pdf"
            report.write_bytes(next(iter(uploaded.values())))
    out = build_submission(notebook, report, saved / "results", saved / "submission.zip")
    files.download(str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="hw2.ipynb", help="completed notebook path")
    parser.add_argument("--report", default="report.pdf", help="written report PDF path")
    parser.add_argument("--no-report", action="store_true", help="omit the report even if a PDF exists")
    parser.add_argument("--results", default="results", help="directory of saved outputs")
    parser.add_argument("--out", default="submission.zip", help="output ZIP path")
    args = parser.parse_args()
    if args.no_report:
        args.report = None
    del args.no_report
    try:
        build_submission(**vars(args))
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
