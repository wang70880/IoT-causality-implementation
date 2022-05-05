#!/bin/sh
cp ./causal_evaluation.py src/causal_evaluation.py
cp ./background_generator.py src/background_generator.py
com="[Testing] Identify a suitable parameter setting: pc_alpha=0.1, alpha_level=0.01, temporal=(3,4)."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
