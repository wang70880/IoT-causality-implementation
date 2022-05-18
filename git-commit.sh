#!/bin/sh
com="[Implementation] Update the architecture of Interaction Miner module. [Testing] Test the updated architecture."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
