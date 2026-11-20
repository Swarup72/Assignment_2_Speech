"""Download the official TESS dataset from Kaggle into data/tess/raw/."""

import inspect
import shutil
from pathlib import Path

import kagglehub


KAGGLE_DATASET_HANDLE = "ejlok1/toronto-emotional-speech-set-tess"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "tess" / "raw"


def download_tess_dataset(target_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download ejlok1/toronto-emotional-speech-set-tess into data/tess/raw/.

    Preserves the exact extracted Kaggle folder structure (including any nested
    duplicate directories) so downstream manifest indexing can inspect and
    deduplicate without deleting raw files.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    existing_wavs = list(target_dir.rglob("*.wav"))
    if len(existing_wavs) >= 2800:
        print(f"[download_tess] Found {len(existing_wavs)} existing .wav files in {target_dir}")
        return target_dir

    sig = inspect.signature(kagglehub.dataset_download)
    if "output_dir" in sig.parameters:
        downloaded_path = Path(
            kagglehub.dataset_download(KAGGLE_DATASET_HANDLE, output_dir=str(target_dir))
        )
    else:
        downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))

    print(f"[download_tess] kagglehub returned path: {downloaded_path}")

    if downloaded_path.resolve() != target_dir.resolve():
        for item in downloaded_path.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    final_wavs = list(target_dir.rglob("*.wav"))
    print(f"[download_tess] Total raw .wav files in {target_dir}: {len(final_wavs)}")
    return target_dir


if __name__ == "__main__":
    download_tess_dataset()
