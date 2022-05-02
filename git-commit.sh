#!/bin/sh
git add -A
git commit -m "Update the debugging printing function."
git push

cd src/tigramite; git add -A; git commit -m "Update the debugging printing function."; git push
