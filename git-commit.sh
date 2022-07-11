#!/bin/sh
com="[Implementation] Finish the construction of golden standard. Next step is to debug and plot the golden standard graph."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
