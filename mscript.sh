#!/bin/bash

for i in {1..5}; do wait python extract_features.py ../mm_ted/data/file_paths_fm.csv & done


