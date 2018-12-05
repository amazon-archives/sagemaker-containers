#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

cp $PWD/$BASEDIR/changehostname.c $2/changehostname.c
sed -ie "s/PLACEHOLDER_HOSTNAME/$1/g" $2/changehostname.c

gcc -o $2/changehostname.o -c -fPIC -Wall $2/changehostname.c
gcc -o $2/libchangehostname.so -shared -export-dynamic $2/changehostname.o -ldl
