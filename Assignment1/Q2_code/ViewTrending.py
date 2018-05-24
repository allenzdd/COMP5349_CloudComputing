'''
edit by Dongdong Zhang 26 April 2018
'''

from pyspark import SparkContext
import argparse
from ml_utils import *

if __name__ == "__main__":
	sc = SparkContext(appName="Impact of Trending on View Number")
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", help="the input path",
                        default='/share/')
	parser.add_argument("--output", help="the output path", 
                        default='/user/dzha4889/out_spark') 
	args = parser.parse_args()

	input_path = args.input
	output_path = args.output

	raw_data = sc.textFile(input_path + "ALLvideos.csv")
	#Get ((video_id, country), (trending_date, views))
	video_all_feature = raw_data.map(extractDateCountryViews)
	#After group and filter(len(date)>2), output is (video_id, country): ((date1, views1), (date2, views2), ...)
	vidCoun_filterGroup = video_all_feature.groupByKey().filter(filterMoreTwoValue)

	#Using map to select first and second day values, then compute percent and filter >= 10
	#Before output change the key to country
	vidCoun_percFilt = vidCoun_filterGroup.map(fir_secDay).filter(filterThousandPrec)

	#Group the country and sort the percentage in values
	coun_sortCounPerc = vidCoun_percFilt.groupByKey().map(sortPerct)

	#Flat the group and integrate the format result
	coun_flat_mat = coun_sortCounPerc.flatMap(flatCounKey)

	coun_flat_mat.saveAsTextFile(output_path)