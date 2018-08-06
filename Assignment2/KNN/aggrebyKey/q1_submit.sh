#!/bin/bash

spark-submit  \
	--master yarn \
	--num-executors $3 \
	--executor-cores $4 \
	--py-files utils.py agg.py \
	--train_input "/share/MNIST/Train-label-28x28.csv" \
	--test_input "/share/MNIST/Test-label-28x28.csv" \
	--k_PCA $1 \
	--k_knn $2 \
	--distance "euclidean" \
	--show_method "accumulator" \
	--repartition_test $5  