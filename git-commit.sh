#!/bin/sh
com="[Testing] Implement the average evaluations about time, truth-count, precision and recall. Prepare to test the code accuracy"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
