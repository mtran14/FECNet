#!/bin/bash
while true
do
	timeout 60 taskset --cpu-list $1 python extract_features.py
done

