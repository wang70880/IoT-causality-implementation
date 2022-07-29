#!/bin/sh
com="[Implementation] Finish the implementation of the data preprocessing module."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
