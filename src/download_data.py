import shutil
from pathlib import Path

import kagglehub

DEST = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    cache_path = Path(kagglehub.dataset_download("nikitricky/chess-positions"))
    DEST.mkdir(parents=True, exist_ok=True)

    for item in cache_path.iterdir():
        target = DEST / item.name
        if not target.exists():
            shutil.copy2(item, target)

    print(f"Data available at {DEST}")


if __name__ == "__main__":
    main()
