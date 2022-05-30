#!/bin/sh
com="[Implementation] Implement the anomaly score threshold calculation and the type 2 anomaly detection. [Testing] Test the accuracy of the threshold calculation function, and the runtime type 2 anomaly detection."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
