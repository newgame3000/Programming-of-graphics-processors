#!/bin/bash

i=0; while [[ $i != 150 ]]; do ./conv.py $i.data $i.png; i=$(($i+1)); done;
