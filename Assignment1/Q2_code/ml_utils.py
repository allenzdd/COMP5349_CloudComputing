'''
edit by Dongdong Zhang 26 April 2018
'''

import re

def extractDateCountryViews(record):
	#input 18 features
	#output ((video_id, country), (trending_date, views))

	try:
		parts = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', record)
		video_id = parts[0]

		trending_date = parts[1]
		#Due to the trending date is string, so can add one by one
		year, day, month = trending_date.split('.')
		date = year + month + day

		views = parts[8]
		country = parts[-1]
		return ((video_id, country), (date, views))
	except:
		return (('video_id', 'country'), ('date', 'views'))


def filterMoreTwoValue(record):
	#Filter numbers of the values--(trending_date, views) more than twos
	keys, values = record
	if len(values) >= 2:
		return (keys, values)


def fir_secDay(record):
	#Using map to select first and second day values, then compute percentage
	#Input ((video_id, country), (date, views))
	#Output (country, (video_id, percentage)) key = country
	keys, values = record
	video_id, country = keys
	firstValues = sorted(values, key = lambda s:int(s[0]))[0][1]
	secondValues = sorted(values, key = lambda s:int(s[0]))[1][1]
	percent = float(int(secondValues) - int(firstValues)) / int(firstValues)
	return (country, (video_id, percent))

def filterThousandPrec(record):
	#Select more than 1000%
	key, values = record
	video_id, percent = values
	if percent >= 10:
		return (key, (video_id, percent))

def sortPerct(record):
	#Sort the percentage in values descending
	try:
		key, values = record
		res = sorted(values, key = lambda v: float(v[1]), reverse=True)
		return (key, res)
	except:
		return ()

def flatCounKey(record):
	#Input (country, (value1, value2, ...))
	#Output (country; video_id, percentage)
	key, values = record
	#flat
	output = ["{}; {}, {:.1%}".format(key, val[0], float(val[1])) for val in values]
	return output















