#!/bin/sh
com="[Implementation] Partially implement security guard module."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
