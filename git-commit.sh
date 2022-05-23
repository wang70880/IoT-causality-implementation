#!/bin/sh
com="[Implementation] Implement the anomaly detection module. Currently it only supports the detection of type-1 attacks for tau=1. [Testing] test the implmentation accuracy and evaluate the detection algorithm's accuracy (for pure testing data and type-1 attack to see the false postive rate)"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
