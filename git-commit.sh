#!/bin/sh
com="[Implementation] Implement the anomaly injection scheme: Case of malicious control; [Debugging] Find out why the accuracy is low."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
