from pathlib import Path
import shutil

from crossfit.dataset.beir.raw import download_all_sampled


current_file_path = Path(__file__).parent.resolve()


if __name__ == "__main__":
    download_all_sampled(out_dir=str(current_file_path))

    # Path to the 'sampled' directory
    sampled_dir = current_file_path / "sampled"

    # Path to the 'beir' directory
    beir_dir = current_file_path / "beir"

    # Remove the 'beir' directory if it already exists
    if beir_dir.exists():
        shutil.rmtree(beir_dir)

    # Rename the 'sampled' directory to 'beir'
    sampled_dir.rename(beir_dir)
