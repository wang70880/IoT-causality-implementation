#!/bin/sh
cp ./causal_evaluation.py src/causal_evaluation.py
cp ./background_generator.py src/background_generator.py
com="[Implementation] Encode the background knowledge into discovery algorithm. [Test] calculate the accuracy using only spatial information"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
