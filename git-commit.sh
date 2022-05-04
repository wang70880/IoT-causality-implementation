#!/bin/sh
cp ./causal_evaluation.py src/causal_evaluation.py
cp ./background_generator.py src/background_generator.py
com="[Testing] Find suitable golden standard parameters"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
