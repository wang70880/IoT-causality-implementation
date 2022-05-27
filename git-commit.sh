#!/bin/sh
com="[Debugging] Analyze the result of the breakpoint detection."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
