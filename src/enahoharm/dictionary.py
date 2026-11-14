"""Variable dictionary (metadata only) extracted from the ENAHO Stata files."""
from pathlib import Path

import pandas as pd

from . import config

COLUMNS = ["year", "module", "name", "label", "value_labels", "dtype", "n_nonmissing"]
MAX_VALUE_LABELS = 300


def _fix_text(s: str) -> str:
    """Repair UTF-8 text that was decoded as latin-1 ("aÃ±o" -> "año")."""
    if not isinstance(s, str):
        return ""
    if "Ã" in s or "Â" in s:
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def _format_value_labels(mapping: dict) -> str:
    if not mapping:
        return ""
    parts = []
    for code, lab in mapping.items():
        code_str = str(int(code)) if float(code).is_integer() else str(code)
        parts.append(f"{code_str}={_fix_text(lab)}")
    text = "; ".join(parts)
    return text if len(text) <= MAX_VALUE_LABELS else text[: MAX_VALUE_LABELS - 3] + "..."


def read_dictionary(dta: Path, year: int, module: str) -> pd.DataFrame:
    with pd.io.stata.StataReader(dta) as reader:
        df = reader.read(convert_categoricals=False)
        var_labels = reader.variable_labels()
        label_sets = reader.value_labels()
        # pandas keeps the variable -> value-label-set link in private attributes.
        lbl_by_var = dict(zip(reader._varlist, reader._lbllist))
    rows = []
    for col in df.columns:
        rows.append(
            {
                "year": year,
                "module": module,
                "name": col.lower(),
                "label": _fix_text(var_labels.get(col, "")),
                "value_labels": _format_value_labels(label_sets.get(lbl_by_var.get(col, ""), {})),
                "dtype": str(df[col].dtype),
                "n_nonmissing": int(df[col].notna().sum()),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def build(raw_dir: Path = config.RAW, out: Path = config.INTERIM / "dictionary.csv") -> pd.DataFrame:
    frames = []
    for year in config.YEARS:
        for module in config.MODULES:
            frames.append(read_dictionary(Path(raw_dir) / str(year) / f"m{module}.dta", year, module))
    d = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, encoding="utf-8")
    return d
