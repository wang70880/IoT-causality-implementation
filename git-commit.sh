#!/bin/sh
com="[Implementation] Implement the anomaly injection scheme: Case of malicious control; [Evaluation] Evaluate the accuracy for malicious control detection."
git add -A
git commit -m "$com"
git push --set-upstream origin breakpoint-theory

cd src/tigramite; git add -A; git commit -m "$com"; git push
