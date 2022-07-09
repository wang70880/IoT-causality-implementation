#!/bin/sh
com="[testing] Test data preprocessing and background generator."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
