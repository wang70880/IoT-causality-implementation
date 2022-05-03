#!/bin/sh
cp ./causal_evaluation.py src/causal_evaluation.py
cp ./background_generator.py src/background_generator.py
git add -A
git commit -m "[Implementation] Implement the background_generator module and update the evaluation module"
git push

cd src/tigramite; git add -A; git commit -m "[Implementation] Implement the background_generator module and update the evaluation module"; git push
