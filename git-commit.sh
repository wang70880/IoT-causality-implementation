#!/bin/sh
com="[Implementation] Update the Bayesian Fitter and implement the query of posterior probability in the Security Guard module. [Testing] test the implmentation accuracy"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
