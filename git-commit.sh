#!/bin/sh
com="[Testing] test the speed and the output of different ARM algorithm; Prepare to compare with association rule mining."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
