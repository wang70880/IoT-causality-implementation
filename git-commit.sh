#!/bin/sh
com="[Implementation] Implement the golden standard constructor."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
