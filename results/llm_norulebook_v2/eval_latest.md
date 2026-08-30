# Eval report (latest)

Train: n=99214, base_rate=0.2630576329953434; Test: n=600, base_rate=0.3717; seed=20260831

| predictor | brier | auc | log_loss | ece | n | base_rate |
|---|---|---|---|---|---|---|
| **base_rate** | 0.1899 | 0.7346 | 0.5618 | 0.0484 | 600 | 0.3717 |
| base_rate / r/AmItheAsshole | 0.2620 | 0.5000 | 0.7174 | 0.1191 | 150 | 0.5467 |
| base_rate / r/legaladvice | 0.1579 | 0.5000 | 0.4982 | 0.0443 | 150 | 0.1933 |
| base_rate / r/personalfinance | 0.1033 | 0.5000 | 0.3647 | 0.0532 | 150 | 0.1133 |
| base_rate / r/unpopularopinion | 0.2365 | 0.5000 | 0.6670 | 0.0656 | 150 | 0.6333 |
| **logistic** | 0.1915 | 0.8201 | 0.5699 | 0.1590 | 600 | 0.3717 |
| logistic / r/AmItheAsshole | 0.1498 | 0.9272 | 0.4716 | 0.2186 | 150 | 0.5467 |
| logistic / r/legaladvice | 0.2127 | 0.5996 | 0.6204 | 0.2255 | 150 | 0.1933 |
| logistic / r/personalfinance | 0.1535 | 0.6333 | 0.4888 | 0.2323 | 150 | 0.1133 |
| logistic / r/unpopularopinion | 0.2499 | 0.4578 | 0.6988 | 0.1423 | 150 | 0.6333 |
| **logistic+isotonic** | 0.1636 | 0.8212 | 0.5015 | 0.0688 | 600 | 0.3717 |
| logistic+isotonic / r/AmItheAsshole | 0.1323 | 0.9310 | 0.4330 | 0.1824 | 150 | 0.5467 |
| logistic+isotonic / r/legaladvice | 0.1480 | 0.5993 | 0.4773 | 0.0792 | 150 | 0.1933 |
| logistic+isotonic / r/personalfinance | 0.1000 | 0.6453 | 0.3477 | 0.0420 | 150 | 0.1133 |
| logistic+isotonic / r/unpopularopinion | 0.2740 | 0.4589 | 0.7479 | 0.1627 | 150 | 0.6333 |
| **llm_oneshot** | 0.1845 | 0.7822 | 0.5505 | 0.0679 | 600 | 0.3717 |
| llm_oneshot / r/AmItheAsshole | 0.2472 | 0.6843 | 0.7096 | 0.1674 | 150 | 0.5467 |
| llm_oneshot / r/legaladvice | 0.1148 | 0.8732 | 0.3753 | 0.1162 | 150 | 0.1933 |
| llm_oneshot / r/personalfinance | 0.0994 | 0.7517 | 0.3420 | 0.1180 | 150 | 0.1133 |
| llm_oneshot / r/unpopularopinion | 0.2768 | 0.7070 | 0.7752 | 0.2686 | 150 | 0.6333 |
| **modcast_agent_norulebook** | 0.1527 | 0.8505 | 0.4715 | 0.0685 | 600 | 0.3717 |
| modcast_agent_norulebook / r/AmItheAsshole | 0.2091 | 0.7703 | 0.6095 | 0.1083 | 150 | 0.5467 |
| modcast_agent_norulebook / r/legaladvice | 0.1272 | 0.7739 | 0.4076 | 0.0610 | 150 | 0.1933 |
| modcast_agent_norulebook / r/personalfinance | 0.0593 | 0.8925 | 0.2322 | 0.0703 | 150 | 0.1133 |
| modcast_agent_norulebook / r/unpopularopinion | 0.2151 | 0.7544 | 0.6367 | 0.1499 | 150 | 0.6333 |

## Paired bootstrap (Brier, same posts; positive = A better)

| A | B | brier A | brier B | advantage A | 95% CI | P(A not better) |
|---|---|---|---|---|---|---|
| base_rate | logistic | 0.1899 | 0.1915 | +0.0015 | [-0.0146, +0.0178] | 0.4287 |
| base_rate | logistic+isotonic | 0.1899 | 0.1636 | -0.0264 | [-0.0395, -0.0132] | 1.0000 |
| base_rate | llm_oneshot | 0.1899 | 0.1845 | -0.0054 | [-0.0292, +0.0182] | 0.6766 |
| base_rate | modcast_agent_norulebook | 0.1899 | 0.1527 | -0.0373 | [-0.0486, -0.0258] | 1.0000 |
| logistic | logistic+isotonic | 0.1915 | 0.1636 | -0.0279 | [-0.0419, -0.0137] | 1.0000 |
| logistic | llm_oneshot | 0.1915 | 0.1845 | -0.0069 | [-0.0311, +0.0176] | 0.7150 |
| logistic | modcast_agent_norulebook | 0.1915 | 0.1527 | -0.0388 | [-0.0541, -0.0234] | 1.0000 |
| logistic+isotonic | llm_oneshot | 0.1636 | 0.1845 | +0.0210 | [-0.0010, +0.0432] | 0.0308 |
| logistic+isotonic | modcast_agent_norulebook | 0.1636 | 0.1527 | -0.0109 | [-0.0274, +0.0059] | 0.9003 |
| llm_oneshot | modcast_agent_norulebook | 0.1845 | 0.1527 | -0.0319 | [-0.0547, -0.0088] | 0.9977 |
