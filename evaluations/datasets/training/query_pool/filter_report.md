# Stage 1 filter report

Started with **810** generated queries.

## Filter pass-through

| Stage | Kept | Dropped | Reasons |
|---|---:|---:|---|
| length floor | 810 | 0 | — |
| type shape | 674 | 136 | type_mismatch_best_practice=40, type_mismatch_comparison=37, type_mismatch_pointer=53, type_mismatch_procedural=6 |
| dedupe (within-type) | 608 | 66 | duplicate_within_best_practice=5, duplicate_within_comparison=8, duplicate_within_pointer=6, duplicate_within_procedural=10, duplicate_within_refusal=37 |
| eval contamination | 599 | 9 | eval_contamination_sim=0.703=1, eval_contamination_sim=0.729=1, eval_contamination_sim=0.742=1, eval_contamination_sim=0.744=1, eval_contamination_sim=0.750=1, eval_contamination_sim=0.761=1, eval_contamination_sim=0.784=1, eval_contamination_sim=0.827=1, eval_contamination_sim=0.838=1 |

Final pool: **599** queries.
