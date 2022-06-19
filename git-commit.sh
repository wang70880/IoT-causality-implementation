#!/bin/sh
com="[Evaluation] Evaluating the detection accuracy."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
