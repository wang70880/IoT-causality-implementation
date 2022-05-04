#!/bin/sh
cp ./causal_evaluation.py src/causal_evaluation.py
cp ./background_generator.py src/background_generator.py
com="[Testing] Set selected_links to be the golden standard and find appropriateparameters (pc_alpha, alpha_level) to maximize the accuracy."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
