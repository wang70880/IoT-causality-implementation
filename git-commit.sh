#!/bin/sh
com="[Debugging] Finish the verification of attribute reduction and type unification. Preprare to verify semantic CI testing results."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
