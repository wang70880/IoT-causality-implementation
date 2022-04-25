#!/bin/sh
git add -A
git commit -m "Update"
git push

cd src/tigramite; git add -A; git commit -m "Update"; git push
