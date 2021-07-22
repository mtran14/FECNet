#!/bin/bash

for i in {0..15}
do
	bash mscript.sh $i &
done
