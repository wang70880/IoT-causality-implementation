#!/bin/sh
com="[Implementation] Implement the InteractionChain class and the Chain Manager module for detections of type-1 attacks; [Test] Test the accuracy of the detections."
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
