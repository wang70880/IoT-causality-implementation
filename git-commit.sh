#!/bin/sh
com="[Debugging] Find out why the detection accuracy of type1 anomalies is low."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
