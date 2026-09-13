# Gold Standard (v6.1)

150 question/answer items over the 12-filing corpus (AAPL, MSFT, AMZN, GOOGL × FY2020/FY2022/FY2024), five strata of 30 each. Every answerable ground truth was recomputed against the SEC source filings; `scripts/validate_gold_standard.py` checks the dataset against the corpus. Methodology: thesis section 3.6.1.

## Files

- `gold_standard_v6.csv` — German source of record, hand-maintained.
- `gold_standard_v6_en.csv` — English version. Identical except for the free-text columns `query`, `gt_value` and `rationale`; this is the file every system and evaluator runs against.

Load them through `GOLD_STANDARD_EN` / `GOLD_STANDARD_DE` in [gold_standard_loader.py](../../src/evaluation/gold_standard_loader.py) rather than spelling out a filename. Corrections are made in the German file, patched cell-wise into the English one, and committed with a message naming the changed ids.

## CSV Schema

Semicolon-delimited (`;`), UTF-8, header row. Columns in file order:

| Column | Type | Description | Example |
|---|---|---|---|
| `id` | int | Stable identifier. | `42` |
| `fa_type` | enum | `FA-1`, `FA-2`, `FA-3`, `FA-4`, `FA-Refusal` (thesis 3.1.1). | `FA-3` |
| `subtype` | enum | Refines `fa_type`, see Subtype Reference. | `growth_cagr` |
| `doc_ids` | string | `\|`-separated document ids `<TICKER>_<YYYY>`; the flat union of `doc_id_groups`. | `AAPL_2024\|MSFT_2024` |
| `doc_id_groups` | string | Acceptable citation sources grouped per required fact. A 10-K reports prior years in comparative columns, so one fiscal year is often carried by several filings. Groups `\|`-separated, members `+`-separated; citing any member satisfies the group. | `AAPL_2020+AAPL_2022\|AAPL_2024` |
| `query` | string | Natural-language question naming company and fiscal year explicitly (except `underspecified` refusal items). | |
| `gt_value` | string | Ground-truth answer, see Notation Convention. Empty on FA-Refusal. | `15.6%` |
| `gt_unit` | enum | `USD_billion`, `USD_million`, `USD_per_share`, `percent`, `pp_change`, `count`, `text`, `n/a`. For composite answers with a company prefix it describes the primary numeric magnitude. | `percent` |
| `source_sections` | string | `\|`-separated section ids holding the source, primary first. Empty on FA-Refusal. | `item_8_income_stmt\|item_7` |
| `expected_answerable` | bool | `false` for the whole FA-Refusal stratum. | `true` |
| `math_type` | enum | FA-3 only: `yoy`, `cagr`, `pp_diff`, `multi_step`, or `none` (qualitative trend); `n/a` elsewhere. | `cagr` |
| `entity_form` | enum | Surface form of the company reference: `name`, `ticker`, `none`. | `ticker` |
| `hypothesis_link` | string | Informational only; hypotheses are mapped by `fa_type`. | `H1` |
| `rationale` | string | One-line reason the item is in the set. | |
| `window_class` | enum | FA-3 only: `within_window` if the year span fits into one report's comparative columns, else `cross_window` (15/15). | `cross_window` |
| `refusal_evidence` | enum | FA-Refusal only, see Refusal Convention. | `silent` |
| `gt_correction` | string | FA-Refusal only: reference text for grading `refusal_quality`. Never an expected answer. | |
| `reference_decomposition` | string | Answerable items only: `\|`-separated sub-questions a complete reasoning chain must cover (completeness dimension). English in both files. | |

## gt_value Notation Convention

- Monetary amounts carry a `$` prefix, also inside composite answers: `$93.7 Mrd`, `GOOGL ($49.3 Mrd)`.
- Non-monetary counts carry no `$`: `AMZN (~1.55 Mio)`.
- Percentages end with `%`: `15.6%`.
- Pairwise FA-4 answers use `WINNER (X op LOSER Y)` with both companies named: `AAPL ($93.7 Mrd > MSFT $88.1 Mrd)`.
- A tilde marks approximate values and is applied symmetrically to both operands.
- Differences carry a sign where the question implies a direction: `+$3.0 Mrd`.

## Answer Format Convention

Fixes definitions and answer formats where ambiguity would cause disagreement between judge, human raters and systems. The instruction in [answer_format_convention.py](../../src/common/answer_format_convention.py) is injected identically into all four systems and must match this section.

- **Ratios and margins**: formula and basis are named in the question or unambiguous. Debt/Equity = Total Liabilities / Total Stockholders' Equity. Margins without a stated segment are consolidated. Always expressed as a percentage, never a decimal fraction.
- **Segment shares**: the question states the denominator explicitly.
- **Trend questions**: the expected answer is the overall direction from the first to the last period in one sentence. Year-by-year values are acceptable supporting evidence and must not be scored as wrong.
- **Growth metrics**: base and end period are the two periods named in the question.

## Subtype Reference

- **FA-1 Single-Fact**: `basis_kpi`, `solvency`, `liquidity`, `eps`
- **FA-2 Cross-Section**: `synthesis`, `segment_share`, `segment_margin`, `ratio_compute`
- **FA-3 Multi-Year**: `growth_yoy`, `growth_cagr`, `pp_change`, `trend_qualitative`, `multi_step_compute`
- **FA-4 Multi-Company**: `cross_firm_pairwise`, `cross_firm_pairwise_growth`, `cross_firm_diff`, `max_min`, `cloud_comparison`
- **FA-Refusal**: `not_in_corpus` (data outside the corpus: other companies, fiscal years or filing types), `false_premise` (question presupposes something untrue), `ambiguous_entity` (company and/or fiscal year not identified)

## Section IDs

`item_1` Business · `item_1a` Risk Factors · `item_7` MD&A · `item_7a` Market Risk · `item_8` Financial Statements · `item_8_income_stmt` · `item_8_balance_sheet` · `item_8_cash_flow` · `item_8_segment`. Item-8 subsections are used whenever the specific statement is known.

## Refusal Convention

A FA-Refusal row has `expected_answerable=false`, empty `gt_value`, `gt_unit=n/a`, empty `source_sections` and `math_type=n/a`. Evaluators branch on `expected_answerable`.

`refusal_evidence` records how far the corpus carries anything beyond "I cannot answer":

| Value | Corpus situation | What a system may legitimately say |
|---|---|---|
| `out_of_scope` | Company, fiscal year or filing type outside the corpus | Name the boundary |
| `counter_evidence` | The filings contradict the premise | Name the contradiction and give the actual figure with its citation |
| `silent` | In scope, but not reported at all | "The filings do not report X", never "X did not happen" |
| `underspecified` | No single company and/or fiscal year identified | Name what is missing |

For `silent` items (e.g. id 132, AMZN dividend; id 137, GOOGL stock split) a confident negative claim would be parametric knowledge, which the stratum exists to detect; the rubric forbids it (`get_evidence_clause` in [custom_evaluator.py](../../src/evaluation/custom_evaluator.py)).

## Stratification

| Stratum | n | Primary use |
|---|---|---|
| FA-1 Single-Fact | 30 | Baseline |
| FA-2 Cross-Section | 30 | H2 |
| FA-3 Multi-Year | 30 | H2; `cross_window` items in H1/H3 |
| FA-4 Multi-Company | 30 | H1, H3 |
| FA-Refusal | 30 | Refusal behaviour (NF-1) |

Answerable strata aim for a roughly even ticker/name split in `entity_form`.
