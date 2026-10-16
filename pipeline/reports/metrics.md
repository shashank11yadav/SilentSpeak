Held-out test speakers: s1, s2, s20, s22, s34 (29997 word segments). A category ships when its top-1 is at least 40%.

| Category | Words | Top-1 | Top-3 | Chance (top-1) | Ships |
|---|---:|---:|---:|---:|:---:|
| command | 3 | 84.9% | 100.0% | 33.3% | yes |
| colour | 3 | 75.6% | 100.0% | 33.3% | yes |
| preposition | 3 | 64.9% | 100.0% | 33.3% | yes |
| letter | 24 | 33.7% | 63.8% | 4.2% | no |
| digit | 9 | 64.4% | 86.3% | 11.1% | yes |
| adverb | 3 | 75.1% | 100.0% | 33.3% | yes |

All 24316 test segments, 45-way: top-1 45.9%, top-5 80.7%.

Few-shot, words never trained on (set, white, with, k, seven, soon): 6-way 5-shot accuracy 78.9% over 1000 episodes (chance 16.7%).
