#!/bin/bash

for i in {0..60}
do
	bash mscript.sh $i &
done
