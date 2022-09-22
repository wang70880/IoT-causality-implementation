#!/bin/sh

git stash
tsocks git pull origin breakpoint-theory
cd src/tigramite; git stash; tsocks git pull origin master
