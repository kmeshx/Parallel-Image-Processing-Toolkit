#!/bin/bash
echo "Running k-means."
#./kmeans images/cs_test1.jpg 3 2048 192 > out.txt
./kmeans >> out.txt
echo "Done."
