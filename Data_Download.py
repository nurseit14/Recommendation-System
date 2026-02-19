import zipfile, io, os, requests, pathlib

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ml-100k"
DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

def main():
    if (DATA_DIR / "u.data").exists():
        print("MovieLens-100K already present:", DATA_DIR)
        return
    print("Downloading MovieLens-100K...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(ROOT / "data")
    print("Extracted to:", ROOT / "data")

if __name__ == "__main__":
    main()