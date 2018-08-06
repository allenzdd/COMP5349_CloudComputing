#!/bin/bash

# for (( i = 2; i < 21; i+=2 )); do
# 	for (( k = 1; k < 9; k++ )); do
# 		spark-submit  \
# 		--master yarn \
# 		--num-executors 4 \
# 		--executor-cores 4 \
# 		RandomForest.py \
# 		--output "spark" \
# 		--numOfTrees $i \
# 		--treeDepth $k
# 		#statements
# 	done
# 	#statements
# done

spark-submit  \
		--master yarn \
		--num-executors 8 \
		--executor-cores 4 \
		RandomForest.py \
		--output "spark" \
		--numOfTrees 5 \
		--treeDepth 6