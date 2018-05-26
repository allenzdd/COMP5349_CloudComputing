#!/usr/bin/python3

'''
edit by Dongdong Zhang 26 April 2018
'''

import sys
import re


def mapper(Coun1, Coun2):
    """ This mapper select tags and return the tag-owner information.
    Input format:  18 features
    Output format: category \t video_id, country
    """
    for line in sys.stdin:
        # Clean input and split it
        parts = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', line)

        if len(parts) < 18:
            continue

        country = parts[-1].strip()

        if country != Coun1 and country != Coun2:
            continue

        if parts[0].strip() != "":
            video_id = parts[0].strip()

        if parts[5].strip() != "":
            category = parts[5].strip()

        print("{}\t{},{}".format(category, video_id, country))


if __name__ == "__main__":
    Coun1 = sys.argv[1]
    Coun2 = sys.argv[2]
    mapper(Coun1, Coun2)
