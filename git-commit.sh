#!/bin/sh
com="[Implementation] Do not compute anomaly scores (as weill as inject anomalies) for attributes which have no interactions (i.e., have 0 incomming degrees or only have automacorrelations)."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
