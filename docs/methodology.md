# Methodology

## Survey and weights

ENAHO is INEI's continuous national household survey, with a stratified, multi-stage area sample. Files
used, per year:

| Module | Content | Unit | Weight used |
|---|---|---|---|
| 02 | Household members | person | `facpob07` (population) |
| 03 | Education | person 3+ | — |
| 05 | Employment and income | person 14+ | `fac500a` (employment/income) |
| 34 | Sumaria (household summary) | household | `factor07` (household) |

Households are identified by `conglome`, `vivienda` and `hogar`, and persons by adding `codperso`. These keys
are read by name for every join, independently of the LLM mapping.

## Poverty

Monetary poverty follows INEI's expenditure method:

- per capita monthly expenditure = `gashog2d / mieperho / 12`;
- a household is **poor** if that value is below `linea` (total poverty line);
- it is **extremely poor** if the value is below `linpe` (food line).

Rates are the share of the **population**: households are weighted by `factor07 × mieperho`. Excluded, and
counted per group, are households with a missing or non-positive weight, a non-positive size, or a missing
expenditure or line.

Two checks:

1. **Against INEI's published national rates.** The tolerance is ±0.5 percentage points.
2. **Against INEI's own `pobreza` flag, weighted the same way.** This separates weighting errors from
   definition errors.

## Area

Area comes from `estrato`: strata 1–5 are urban and 6–8 rural (INEI convention).

## Labour

All labour rates use `fac500a` over the population aged 14 and over, INEI's working-age population (PET).

- **Employment rate:** persons with `ocu500 == 1` (employed) as a share of the population aged 14+.
- **Informality rate:** persons with `ocupinf == 1` as a share of the employed. It is available for 2019 only
  (see README, Limitations).
- **Labour income:** the sum of the four imputed annual monetary components: dependent and independent work,
  each in the main and in secondary jobs. Missing components count as zero.

## Years of schooling (approximation)

The variables are `p301a` (level attained) and either `p301b` (last year completed) or `p301c` (last grade
completed). Peru has 6 years of primary and 5 of secondary.

| `p301a` | Level | Years |
|---|---|---|
| 1–2 | No level, pre-school | 0 |
| 3 | Incomplete primary | last year or grade (max 6) |
| 4 | Complete primary | 6 |
| 5 | Incomplete secondary | 6 + last year |
| 6 | Complete secondary | 11 |
| 7–10 | Higher, non-university or university | 11 + last year |
| 11 | Postgraduate | 16 + last year |
| 12 | Special basic education | missing |

## LLM settings

**Request settings:**

- model `claude-haiku-5-5`, default adaptive thinking, effort `medium`;
- the answer is constrained to a JSON schema whose target field is an enum of the schema names plus `none`;
- the model does not accept a temperature setting.

**Reproducibility.** Every answer is stored in `data/llm_cache/` with its token usage, keyed by a hash of the
model and the full prompt. The pipeline reproduces from that cache without calling the API.

**Malformed answers.** These cases are marked `invalid` and sent to human review:

- unparseable JSON;
- a target outside the schema;
- a confidence outside [0, 1];
- a variable missing from the answer.
