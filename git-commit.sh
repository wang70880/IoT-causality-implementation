#!/bin/sh
com="[Implementation] Update Bayesian Fitter module."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
