#!/bin/sh
com="[Debugging] Preprare to implement the data debugger, and use new dataset (hh130) to verify the preprocessing process and verify semantic CI testing results."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
