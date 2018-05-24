#!/bin/bash

spark-submit  \
  --master local[2] \
  --num-executors 3 \
  --py-files ml_utils.py ViewTrending.py \
   --input "/share/" \
   --output "/user/dzha4889/out_spark"