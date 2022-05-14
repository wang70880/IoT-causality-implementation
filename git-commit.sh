#!/bin/sh
com="[Testing] Testing the bayesian predictor module: the correctness of material transformation functionality"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
