#!/bin/sh
git add -A
git commit -m "Optimize the shuffle test + CMI computation by using numba. 50% speedup is achieved."
git push

cd src/tigramite; git add -A; git commit -m "Optimize the shuffle test + CMI computation by using numba. 50% speedup is achieved."; git push
