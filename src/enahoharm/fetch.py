"""Download ENAHO modules from INEI and keep only the main Stata file of each."""
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from . import config


def module_url(code: int, module: str) -> str:
    return f"{config.BASE_URL}/{code}-Modulo{module}.zip"


def _main_member(z: zipfile.ZipFile) -> str:
    # INEI zips may ship a "-12g" variant (12 geographic groups) next to the base file.
    dtas = [i for i in z.infolist() if i.filename.lower().endswith(".dta") and "-12g" not in i.filename.lower()]
    if not dtas:
        raise ValueError("no .dta file in archive")
    return max(dtas, key=lambda i: i.file_size).filename


def main_dta(zip_path: Path, year: int, module: str) -> str:
    with zipfile.ZipFile(zip_path) as z:
        return _main_member(z)


def target_path(year: int, module: str, raw_dir: Path) -> Path:
    return Path(raw_dir) / str(year) / f"m{module}.dta"


def download(year: int, module: str, raw_dir: Path, opener=urllib.request.urlopen) -> Path:
    """Return the path of the module's main .dta, downloading it only if missing."""
    out = target_path(year, module, raw_dir)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = module_url(config.YEARS[year], module)
    with opener(url, timeout=300) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        member = _main_member(z)
        tmp = out.with_suffix(".part")
        with z.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
    tmp.replace(out)
    return out


def fetch_all(raw_dir: Path = config.RAW) -> list[Path]:
    paths = []
    for year in config.YEARS:
        for module in config.MODULES:
            p = download(year, module, raw_dir)
            print(f"{year} m{module}: {p.stat().st_size / 1e6:.1f} MB")
            paths.append(p)
    return paths
