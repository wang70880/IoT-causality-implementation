#!/bin/sh
com="[Debugging] Find out the reason why the number of false positives of score estimation is high."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
