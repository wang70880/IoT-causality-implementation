#!/bin/sh
com="[Implementation] Prepare to implement the type-1 anomaly injection; [Test] Test the detection of type-1 anomalies without injections."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
