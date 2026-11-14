"""Prompt text for mapping survey variables to the target schema."""
import pandas as pd

from .schema import Target

INSTRUCTIONS = """You harmonize variables from ENAHO, Peru's national household survey (INEI), across years.

You receive a batch of source variables from one survey year and module: name, label (in Spanish) and value
labels. For each source variable, decide which TARGET variable of the harmonized schema it measures, or
"none" if it matches no target.

Rules:
- Map a source variable to a target only if it measures exactly that concept, unit and reference period.
  Similar but different concepts (a sub-component, a different job, a flag about imputation, a variant
  computed with another methodology) are "none".
- A target is usually matched by one source variable per module; identifiers and design variables (cluster,
  dwelling, household, stratum, weights) can appear in every module.
- Use the label and value labels, not only the name; names can change between years.
- confidence is your probability (0 to 1) that the mapping is correct.
- justification: one short sentence in English.
- Return one entry for every source variable in the batch, using its exact name.
"""


def system_prompt(targets: list[Target]) -> str:
    lines = [INSTRUCTIONS, "TARGETS (name | unit | description | expected values):"]
    for t in targets:
        if t.kind == "direct":
            lines.append(f"- {t.name} | {t.unit} | {t.description} | {t.values}")
    return "\n".join(lines)


def render(batch: pd.DataFrame) -> str:
    first = batch.iloc[0]
    lines = [f"Survey year {int(first['year'])}, module {str(first['module']).zfill(2)}. Source variables:"]
    for _, r in batch.iterrows():
        vl = f" | values: {r['value_labels']}" if isinstance(r["value_labels"], str) and r["value_labels"] else ""
        lines.append(f"- {r['name']} | {r['label']}{vl}")
    return "\n".join(lines)


def answer_schema(targets: list[Target]) -> dict:
    allowed = sorted(t.name for t in targets if t.kind == "direct") + ["none"]
    return {
        "type": "object",
        "properties": {
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "target": {"type": "string", "enum": allowed},
                        "confidence": {"type": "number"},
                        "justification": {"type": "string"},
                    },
                    "required": ["name", "target", "confidence", "justification"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["mappings"],
        "additionalProperties": False,
    }
