#!/bin/sh
com="[Implementation] Fix the wrong updates of the phantom state machine; [Debugging] Find out the reason why the accuracy of score estimation is so low."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
