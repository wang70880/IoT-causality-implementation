#!/bin/sh
com="[Implementation] Prepare to implement the InteractionChain class for the anomaly detection module."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
