#!/bin/bash

spark-submit  \
	--master yarn \
	--num-executors $1 \
	--executor-cores $2 \
	--py-files q3_utils.py q3.py \
	--train_input "/share/MNIST/Train-label-28x28.csv" \
	--test_input "/share/MNIST/Test-label-28x28.csv" \
	--maxIter 10 \
	--layers "784,50,10" \
	--blockSize $3 \