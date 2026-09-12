# MumGuard locked signal challenge — execution summary

The frozen `mumguard-signal-locked-v1` protocol was executed on the nine hash-verified private client originals after the protocol/runner had been committed. No disease head was fitted and no threshold was selected.

## Result

| Frozen signal | Tumor-bearing subjects above `CLIENT-MOUSE-0038` | Median positive-minus-negative |
| --- | ---: | ---: |
| Core hyperthermia-like local elevation | **7 / 8** | **+0.175464** |
| Static frozen composite | **6 / 8** | **+0.065358** |
| Abnormal-skin unilateral pattern score | **0 / 8** | **-0.107609** |
| Generic global anomaly score | 0 / 8 (4 ties) | -0.013455 |

The strongest new label-free signal in this bounded challenge is therefore the **directed focal/core thermal-elevation channel**. It ranked seven of the eight tumor-bearing mouse captures above the sole no-tumor capture without fitting a disease classifier.

The pre-frozen static composite ranked six of eight tumor-bearing captures above the sole no-tumor capture. The unilateral generic abnormal-pattern formulation moved in the wrong direction on this cohort and must **not** be silently reweighted after seeing these labels. It remains an engineering measurement channel and should be redesigned/evaluated in a new locked experiment, preferably at the full bilateral human-session level where abnormal skin behaviour can be compared against the contralateral breast.

## Interpretation for the architecture

This result supports keeping the system signal-first rather than replacing it with another raw-image classifier:

- focal local thermal comparison is producing useful development separation;
- generic whole-field anomaly magnitude is not sufficient by itself;
- the full human target still adds two pieces that the nine single mouse captures cannot exercise: multi-frame side reconstruction and true LEFT-vs-RIGHT internal-control comparison;
- DINO/local visual evidence remains a complementary channel from PR #41 rather than a substitute for the TLC measurement engine.

This is a development ranking on 8 positive / 1 negative previously exposed subjects, not independent validation and not a cancer probability.
