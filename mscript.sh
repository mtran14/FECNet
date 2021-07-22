#!/bin/bash
while true
do
	timeout 600 taskset --cpu-list $1 python extract_features.py
done

