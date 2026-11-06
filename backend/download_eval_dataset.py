import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REAL_DIR = BASE_DIR / "dataset" / "real"
FAKE_DIR = BASE_DIR / "dataset" / "fake"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

# 10 High quality public domain / CC portraits with diverse age, gender, ethnicity
REAL_SOURCES = [
    ("real_obama.jpg", "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"),
    ("real_harris.jpg", "https://upload.wikimedia.org/wikipedia/commons/4/41/Kamala_Harris_Vice_Presidential_Portrait.jpg"),
    ("real_biden.jpg", "https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg"),
    ("real_lincoln.jpg", "https://upload.wikimedia.org/wikipedia/commons/4/44/Abraham_Lincoln_head_on_shoulders_photo_portrait.jpg"),
    ("real_watson.jpg", "https://upload.wikimedia.org/wikipedia/commons/7/7f/Emma_Watson_2013.jpg"),
    ("real_musk.jpg", "https://upload.wikimedia.org/wikipedia/commons/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg"),
    ("real_huang.jpg", "https://upload.wikimedia.org/wikipedia/commons/b/b3/Jensen_Huang_2024.jpg"),
    ("real_nadella.jpg", "https://upload.wikimedia.org/wikipedia/commons/7/78/MS-Exec-Cadre-Satya-Nadella-1_0.jpg"),
    ("real_curie.jpg", "https://upload.wikimedia.org/wikipedia/commons/c/c8/Marie_Curie_c._1920s.jpg"),
    ("real_einstein.jpg", "https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg")
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def download_file(url, target_path):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read()
        target_path.write_bytes(content)

def main():
    print("=== Downloading Real Portraits from Wikimedia ===")
    for filename, url in REAL_SOURCES:
        target = REAL_DIR / filename
        if target.exists() and target.stat().st_size > 1000:
            print(f"Already exists: {filename}")
            continue
        try:
            print(f"Downloading {filename}...")
            download_file(url, target)
            print(f"Saved {filename} ({target.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
        time.sleep(1)

    print("\n=== Downloading Synthetic Faces from ThisPersonDoesNotExist ===")
    for i in range(1, 11):
        filename = f"synthetic_face_{i:02d}.jpg"
        target = FAKE_DIR / filename
        if target.exists() and target.stat().st_size > 1000:
            print(f"Already exists: {filename}")
            continue
        try:
            print(f"Downloading {filename}...")
            # Querying the endpoint produces a new StyleGAN2 synthetic image
            download_file("https://thispersondoesnotexist.com", target)
            print(f"Saved {filename} ({target.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
        time.sleep(1.5)

    print("\nDataset preparation completed.")
    print(f"Total REAL images: {len(list(REAL_DIR.glob('*')))}")
    print(f"Total FAKE images: {len(list(FAKE_DIR.glob('*')))}")

if __name__ == "__main__":
    main()

