#!/bin/sh
cp ./causal_evaluation.py src/causal_evaluation.py
git add -A
git commit -m "Debugging."
git push

cd src/tigramite; git add -A; git commit -m "Debugging."; git push
