# network_accuracy_honesty

**Question:** what does adding cameras actually buy, and what does it cost?

**Answer:** on six frozen drives, the five-camera network is more accurate than the best
single camera (3.34 vs 3.56 cm median) and 9.3 points less honest (77.1% vs 86.4% coverage
of a nominal 95% ellipse). Score-conditioned covariance is the most accurate arm at 2.31 cm
and still only 83.1% honest. Only an explicit persistent-bias state reaches 91.8%, costing
+0.49 cm of median error.

This is the thesis's central table. The mechanism behind the honesty column lives in
`experiments/reading_independence/`.

`summarize_network.py` reads the frozen six-run selection and reports accuracy, honesty and
correction accounting together. It fits nothing and launches nothing.

Results, interpretation and limitations: `logs/studies/network_accuracy_honesty_20260907/`.
