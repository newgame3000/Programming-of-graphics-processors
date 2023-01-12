#!/bin/bash

make
cd res
rm *res.gif
cd ..
./so* --default > 2.test
./so* < 2.test
cd res
./run.sh
rm *.data
convert `ls --sort=t *.png` res.gif
rm *.png
