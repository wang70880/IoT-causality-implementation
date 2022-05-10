#!/bin/sh
com="[Evaluation] Finish the evaluation of discovery accuracy; Prepare to compare with association rule mining."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
