"""README results block and figures."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
INK, MUTED, ACCENT, ACCENT2 = "#1f2933", "#7b8794", "#2563eb", "#d97706"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _review_summary(review: pd.DataFrame) -> dict:
    decided = review["decision"].fillna("") != ""
    return {
        "queued": len(review),
        "decided": int(decided.sum()),
        "changed": int(review["decision"].isin(["correct", "reject"]).sum()),
        "by_decision": review.loc[decided, "decision"].value_counts().to_dict(),
    }


def results_markdown(metrics: dict, benchmark: pd.DataFrame, review: pd.DataFrame) -> str:
    r = _review_summary(review)
    lines = []
    if r["decided"] == 0:
        pending = ("> **Status: LLM proposal, human review pending.** The mappings below have not yet been "
                   "checked by a person.")
        lines += [pending, ""]
    lines += [
        "**LLM mapping** (Claude Haiku 5.5, every variable of modules 02, 03, 05 and 34 in both years)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Source variables mapped | {metrics['n_variables']:,} |",
        f"| Variables that belong to the schema (ground truth) | {metrics['n_mapped_truth']:,} |",
        f"| Accuracy on those variables (right target) | {_pct(metrics['accuracy_mapped'])} |",
        f"| Recall (recognised as mappable) | {_pct(metrics['recall_mapped'])} |",
        f"| Precision of the mapped calls | {_pct(metrics['precision_mapped'])} |",
        f"| Accuracy on out-of-schema variables (`none`) | {_pct(metrics['accuracy_none'])} |",
        f"| Invalid answers | {metrics['n_invalid']} |",
        f"| Tokens (input / output) | {metrics['input_tokens']:,} / {metrics['output_tokens']:,} |",
        f"| Cost | USD {metrics['cost_usd']:.2f} |",
        "",
        (f"**Human review:** {r['queued']} variables routed to review, {r['decided']} decided, "
         f"{r['changed']} changed by the reviewer (corrected or rejected)."),
        "",
        "**Poverty benchmark against INEI** (population-weighted, % of population; tolerance ±0.5 pp)",
        "",
        "| Year | Measure | This pipeline | INEI | Difference (pp) | Within tolerance |",
        "|---|---|---|---|---|---|",
    ]
    for _, b in benchmark.iterrows():
        lines.append(f"| {int(b['year'])} | {b['measure']} | {b['ours']:.2f} | {b['inei']} | {b['diff']:+.2f} | "
                     f"{'yes' if b['within_tol'] else 'no'} |")
    return "\n".join(lines)


def update_readme(readme: Path, metrics: dict, benchmark: pd.DataFrame, review: pd.DataFrame) -> None:
    text = Path(readme).read_text(encoding="utf-8")
    block = f"{START}\n{results_markdown(metrics, benchmark, review)}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, count=1, flags=re.DOTALL)
    Path(readme).write_text(new, encoding="utf-8")


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def figures(metrics: dict, poverty: pd.DataFrame, review: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    cal = pd.DataFrame(metrics["calibration"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(cal["bucket"], cal["accuracy"].fillna(0) * 100, color=ACCENT)
    for i, (_, c) in enumerate(cal.iterrows()):
        ax.text(i, (c["accuracy"] or 0) * 100 + 2, f"n={c['n']}", ha="center", fontsize=8, color=MUTED)
    ax.set_ylim(0, 110)
    ax.set_xlabel("LLM confidence", color=INK)
    ax.set_ylabel("Accuracy (%)", color=INK)
    _style(ax, "Accuracy of the LLM mapping by stated confidence")
    paths.append(out_dir / "llm_calibration.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    p = poverty[poverty["area"].isin(["urban", "rural"])]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    years = sorted(p["year"].unique())
    width = 0.38
    for k, (area, color) in enumerate([("urban", ACCENT), ("rural", ACCENT2)]):
        vals = [p[(p["year"] == y) & (p["area"] == area)]["poverty"].iloc[0] for y in years]
        xs = [i + (k - 0.5) * width for i in range(len(years))]
        ax.bar(xs, vals, width, label=area, color=color)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_ylabel("Poverty rate (% of population)", color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Monetary poverty by area, ENAHO (recomputed with sampling weights)")
    paths.append(out_dir / "poverty_area.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    r = review.copy()
    r["decision"] = r["decision"].fillna("").replace("", "pending")
    counts = r.groupby(["reason", "decision"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    left = pd.Series(0, index=counts.index)
    palette = {"accept": ACCENT, "correct": ACCENT2, "reject": "#b91c1c", "pending": "#cbd2d9"}
    for dec in [d for d in ["accept", "correct", "reject", "pending"] if d in counts]:
        ax.barh(counts.index, counts[dec], left=left, label=dec, color=palette[dec])
        left += counts[dec]
    ax.set_xlabel("Variables", color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Human review queue: why each variable was routed, and the decision")
    paths.append(out_dir / "review_outcomes.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths
