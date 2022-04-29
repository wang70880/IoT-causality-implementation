#!/bin/sh
git add -A
git commit -m "Identify the problem of histogram computation casued by high dimensionality. Ad-hoc way is to set max_conds_dim(px/py) to avoid it"
git push

cd src/tigramite; git add -A; git commit -m "Identify the problem of histogram computation casued by high dimensionality. Ad-hoc way is to set max_conds_dim(px/py) to avoid it"; git push
