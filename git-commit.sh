#!/bin/sh
cp ./causal_evaluation.py src/causal_evaluation.py
git add -A
git commit -m "[Testing] Increae max_conds_dim to 10 and maximum_comb to 5. Supposedly it should reduce the number of false positives"
git push

cd src/tigramite; git add -A; git commit -m "[Testing] Increae max_conds_dim to 10 and maximum_comb to 5. Supposedly it should reduce the number of false positives"; git push
