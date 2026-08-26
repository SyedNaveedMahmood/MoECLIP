"""Download all MoECLIP datasets from Google Drive and extract into data/."""
import os
import sys
import zipfile
import subprocess

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "_downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# name -> google drive file id (from README)
DATASETS = {
    "MVTec": "1JkzLzwP4-sGHkPubeQplazX-CizuuaEw",
    "VisA": "1kNn07-KcISquckAm209ZIecGGrjdi8Ph",
    "BTAD": "1nB3wlkHKUiLpMJFCqCxgGTgkN0kcfgLp",
    "RSDD": "1EiYoVE9weICu4N66bscZp8dFpyE0a3LC",
    "DTD-Synthetic": "1Ej1LUiTLB6e55EXVS7HR6VIP0IDEcR-S",
    "Brain": "1xPuHEenVPAyZe7ZsQElACkY6QBmiDQ1E",
    "headct": "1bxv22XWNqY7D4JdbANtPauDxkEiVlBLy",
    "Liver": "1AfelY6jIde5pn6YblyRhRif5net_WLVp",
    "Retina": "1UH2QtPy9M9-U8SaPVxZJTBSzSWLbfAkm",
    "Colon_colonDB": "1hlRejL0XHxBFVy0xf8RRg9vpdSLqGPBm",
    "Colon_clinicDB": "1bgcfV2Fjpe5YhDe78zHNFqagImF-TNqy",
    "Colon_cvc300": "1u0QHmoeCP0nKYVB0CHAuxnQDResXQmhP",
    "Endo": "1ixNCD7VH10reO18L685UZh_kZjXVhz_m",
    "Colon_Kvasir": "1hvrXW8uOo8_UuOKL-SjurhwRYRI8Bekq",
}


def download(name: str, file_id: str) -> str | None:
    out = os.path.join(DOWNLOAD_DIR, f"{name}.zip")
    if os.path.exists(out):
        print(f"[skip] {out} already downloaded")
        return out
    cmd = [sys.executable, "-m", "gdown", file_id, "-O", out]
    print("[cmd]", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"[ERROR] download failed for {name}")
        return None
    return out


def extract(zip_path: str, name: str):
    dest = os.path.join(DATA_DIR, name)
    marker = os.path.join(dest, ".extracted")
    if os.path.exists(marker):
        print(f"[skip] {dest} already extracted")
        return
    print(f"[extract] {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    open(marker, "w").close()
    print(f"[done] {name}")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(DATASETS.keys())
    for name in targets:
        fid = DATASETS[name]
        print(f"\n===== {name} =====")
        zp = download(name, fid)
        if zp:
            try:
                extract(zp, name)
            except zipfile.BadZipFile:
                print(f"[WARN] {zp} is not a zip; leaving as-is in _downloads/")