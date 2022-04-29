#!/bin/sh
git add -A
git commit -m "Update the golden standard construction for user correlations; going to evaluate the accuracy and identfy why there are FNs."
git push

cd src/tigramite; git add -A; git commit -m "Update the golden standard construction for user correlations; going to evaluate the accuracy and identfy why there are FNs."; git push
