#!/bin/sh
com="[Implementation] Update the partition_config parameter (from tuple to int); Prepare to initiate accuracy evaluations about the partition_config params"
git add -A
git commit -m "$com"
git push

cd src/tigramite; git add -A; git commit -m "$com"; git push
