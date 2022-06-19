#!/bin/sh
com="[Evaluation] Evaluating the detection for 100 single point anomalies."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
