#!/bin/sh
com="[Testing] Update the evaluation module and test the accuracy of the program; Prepare to compare with association rule mining."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
