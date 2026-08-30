# Eval report (latest)

Train: n=99214, base_rate=0.2630576329953434; Test: n=1000, base_rate=0.3620; seed=20260831

| predictor | brier | auc | log_loss | ece | n | base_rate |
|---|---|---|---|---|---|---|
| **base_rate** | 0.1904 | 0.7295 | 0.5634 | 0.0369 | 1000 | 0.3620 |
| base_rate / r/AmItheAsshole | 0.2558 | 0.5000 | 0.7049 | 0.0764 | 250 | 0.5040 |
| base_rate / r/legaladvice | 0.1542 | 0.5000 | 0.4889 | 0.0390 | 250 | 0.1880 |
| base_rate / r/personalfinance | 0.1131 | 0.5000 | 0.3883 | 0.0386 | 250 | 0.1280 |
| base_rate / r/unpopularopinion | 0.2386 | 0.5000 | 0.6715 | 0.0709 | 250 | 0.6280 |
| **logistic** | 0.1937 | 0.8184 | 0.5735 | 0.1664 | 1000 | 0.3620 |
| logistic / r/AmItheAsshole | 0.1610 | 0.9421 | 0.4956 | 0.2395 | 250 | 0.5040 |
| logistic / r/legaladvice | 0.2059 | 0.6283 | 0.6010 | 0.2171 | 250 | 0.1880 |
| logistic / r/personalfinance | 0.1597 | 0.6087 | 0.5019 | 0.2185 | 250 | 0.1280 |
| logistic / r/unpopularopinion | 0.2480 | 0.4717 | 0.6957 | 0.1146 | 250 | 0.6280 |
| **logistic+isotonic** | 0.1617 | 0.8194 | 0.4970 | 0.0580 | 1000 | 0.3620 |
| logistic+isotonic / r/AmItheAsshole | 0.1242 | 0.9453 | 0.4148 | 0.1893 | 250 | 0.5040 |
| logistic+isotonic / r/legaladvice | 0.1460 | 0.6260 | 0.4684 | 0.0600 | 250 | 0.1880 |
| logistic+isotonic / r/personalfinance | 0.1105 | 0.6205 | 0.3752 | 0.0199 | 250 | 0.1280 |
| logistic+isotonic / r/unpopularopinion | 0.2660 | 0.4734 | 0.7295 | 0.1288 | 250 | 0.6280 |

## Paired bootstrap (Brier, same posts; positive = A better)

| A | B | brier A | brier B | advantage A | 95% CI | P(A not better) |
|---|---|---|---|---|---|---|
| base_rate | logistic | 0.1904 | 0.1937 | +0.0032 | [-0.0095, +0.0156] | 0.3060 |
| base_rate | logistic+isotonic | 0.1904 | 0.1617 | -0.0288 | [-0.0384, -0.0189] | 1.0000 |
| logistic | logistic+isotonic | 0.1937 | 0.1617 | -0.0320 | [-0.0425, -0.0208] | 1.0000 |
