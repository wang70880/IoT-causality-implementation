#!/bin/sh
com="[Implementation] Prepare to implement the InteractionChain class and the Chain Manager module for detections of type-2 attacks; [Test] Test why bayesian fitting is slow."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
