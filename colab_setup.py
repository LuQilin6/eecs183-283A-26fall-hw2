"""Colab dependency installation and persistent output storage."""
import importlib.metadata
import subprocess
import sys
from pathlib import Path


def setup():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Select Runtime > Change runtime type > Runtime Version > 2026.07 "
            "(Python 3.12), then run the setup cell again."
        )

    from packaging.requirements import Requirement
    from packaging.utils import parse_wheel_filename

    root = Path(__file__).resolve().parent
    # Colab supplies its own CUDA-enabled PyTorch and notebook infrastructure.
    supplied = {"torch", "jupyterlab", "ipykernel", "nbconvert"}
    requirements = []
    expected = {}
    for filename in ("requirements.txt", "requirements-models.txt"):
        for line in (root / filename).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            requirement = Requirement(line)
            if requirement.name in supplied:
                continue
            requirements.append(line)
            if requirement.url:
                version = str(parse_wheel_filename(requirement.url.rsplit("/", 1)[-1])[1])
            else:
                version = next(iter(requirement.specifier)).version
            expected[requirement.name] = version

    def installed(name, version):
        try:
            return importlib.metadata.version(name) == version
        except importlib.metadata.PackageNotFoundError:
            return False

    if not all(installed(name, version) for name, version in expected.items()):
        print("Installing homework libraries and language models (several minutes)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", *requirements],
            check=True,
        )
        raise RuntimeError(
            "Installation finished. Choose Runtime > Restart session, then run "
            "this setup cell again. This is needed once per new runtime."
        )
    for name, version in expected.items():
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, "__version__", version) != version:
            raise RuntimeError("Choose Runtime > Restart session, then rerun setup.")

    from google.colab import drive

    drive.mount("/content/drive")
    saved = Path("/content/drive/MyDrive/cs183-hw2")
    (saved / "results").mkdir(parents=True, exist_ok=True)
    results = root / "results"
    if results.is_symlink():
        if results.resolve() != (saved / "results").resolve():
            raise RuntimeError(f"Unexpected results link: {results}. Check its destination.")
    elif results.exists():
        raise RuntimeError(
            f"{results} already exists. Back it up and move it out of the repository, "
            "then rerun setup to connect Google Drive."
        )
    else:
        results.symlink_to(saved / "results", target_is_directory=True)
    print(f"Setup ready. Results are saved in {saved / 'results'}.")
