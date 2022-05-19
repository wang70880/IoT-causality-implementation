#!/bin/sh
com="[Implementation] Seperate testing data and training data. [Testing] Test the discovery accuracy on training data and the implementation accuracy of data seperation."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
