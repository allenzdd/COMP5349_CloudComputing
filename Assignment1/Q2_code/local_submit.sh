#!/bin/bash

spark-submit  \
  --master yarn \
  --num-executors 3 \
  --py-files ml_utils.py ViewTrending.py \
   --input "/share/" \
   --output "/user/dzha4889/out_spark"