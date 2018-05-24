#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Invalid number of parameters!"
    echo "Usage: ./driver.sh [input_location] [output_location] [country_1] [country_2]"
    exit 1
fi

hadoop jar  $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar \
-D mapreduce.job.reduces=3 \
-D mapreduce.job.name='country inverted list' \
-file mapper.py \
-mapper "mapper.py $3 $4" \
-file reducer.py \
-reducer "reducer.py $3 $4" \
-input $1 \
-output $2
