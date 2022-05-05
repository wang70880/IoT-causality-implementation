#!/bin/sh
com="[Testing] Test the run script for evalautions of partition accuracy"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
