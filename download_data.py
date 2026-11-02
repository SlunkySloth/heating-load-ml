"""Download the UCI Energy Efficiency dataset into data/.

Source: Tsanas & Xifara (2012), UCI Machine Learning Repository, dataset 242.
This is a stand-in dataset. It is not "usim" data.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
FILE_NAME = "ENB2012_data.xlsx"
DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    target = DATA_DIR / FILE_NAME
    if target.exists():
        print(f"Already downloaded: {target}")
        return

    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    print(f"Received {len(payload) / 1024:.0f} KB")

    # Read only the one expected member, so nothing else in the zip reaches disk.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        content = archive.read(FILE_NAME)

    DATA_DIR.mkdir(exist_ok=True)
    target.write_bytes(content)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
