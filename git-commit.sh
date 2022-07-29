#!/bin/sh
com="[Breakpoint] Prepare to update the shell script and the parameter setting: For evaluation purpose."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
