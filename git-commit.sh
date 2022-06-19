#!/bin/sh
com="[Debugging] Reason for the high false positive of devices M001."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
