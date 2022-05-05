#!/bin/sh
com="[Evaluation] Update the run script"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
