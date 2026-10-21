# Evidence faithfulness (deletion test, temperature-calibrated probabilities)

- 210 probe issues; 178 explanations contained quotes; 93.1% of 331 quotes occur verbatim in the issue.
- Explained label equals the scored label in 92.9% of issues.
- Deletion test on 164 issues with deletable quotes (mean 130 of 921 body characters removed):
  - mean drop in the predicted label's probability: evidence 0.025, random 0.014 (Wilcoxon p = 0.115)
  - label flips: evidence 6.1%, random 3.0%
  - evidence deletion lowers confidence more than random deletion in 55.5% of issues
