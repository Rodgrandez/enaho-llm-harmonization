"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"            # git-ignored: INEI microdata
INTERIM = DATA / "interim"    # committed: metadata only (dictionary, mappings)
PROCESSED = DATA / "processed"  # git-ignored: harmonized microdata
CACHE = DATA / "llm_cache"    # committed: raw LLM answers
SCHEMA = ROOT / "schema" / "target.yaml"
VALIDATION = ROOT / "validation"
REPORTS = ROOT / "reports"

# INEI survey codes per year (verified 2026-11-14 against the sumaria files).
YEARS = {2019: 687, 2024: 966}
# 02 members, 03 education, 05 employment, 34 sumaria (household summary).
MODULES = ["02", "03", "05", "34"]

BASE_URL = "https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA"
