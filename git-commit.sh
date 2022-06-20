#!/bin/sh
com="[Implementation] Implement the anomaly injection scheme: Case of malicious control; [Testing] Test the accuracy of the implementation."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
