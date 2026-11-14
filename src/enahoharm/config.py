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

# INEI official national monetary poverty rates, % of population (Informe Tecnico "Evolucion de la Pobreza
# Monetaria"; 2019 report published 2020, 2024 report published May 2025).
INEI_OFFICIAL = {
    2019: {"poverty": 20.2, "extreme_poverty": 2.9},
    2024: {"poverty": 27.6, "extreme_poverty": 5.5},
}
BENCHMARK_TOLERANCE_PP = 0.5
