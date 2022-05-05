#!/bin/sh
com="[Evaluation] Evaluate the accuracy of discovery given different partitions and background knowledge"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
