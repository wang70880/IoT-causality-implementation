#!/bin/sh
com="[Debugging] Find out why the detection accuracy of type1 anomalies is low."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
