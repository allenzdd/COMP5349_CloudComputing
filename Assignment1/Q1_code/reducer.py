#!/usr/bin/python3

'''
edit by Dongdong Zhang 26 April 2018
'''

import sys


def read_map_output(file):
    """ Return an iterator for key, value pair extracted from file (sys.stdin).
    Input format:  category \t video_id, country
    Output format: category, (video_id, country)
    """
    for line in file:
        if len(line.strip().split("\t")) != 2:
            continue
        yield line.strip().split("\t")


def reducer(Coun1, Coun2):

    coun_cat_videoid = {}

    for category, video_id_country in read_map_output(sys.stdin):
        video_id, country = video_id_country.strip().split(',')

        # Create 3d dictionary 3rd feature is a [], can add string value
        if category not in coun_cat_videoid:
            coun_cat_videoid[category] = {}
            coun_cat_videoid[category][Coun1] = []
            coun_cat_videoid[category][Coun2] = []

        coun_cat_videoid[category][country].append(video_id)

    for cat in coun_cat_videoid:
        # Clear the set and overlap count
        set_coun1 = set()
        set_coun2 = set()
        overlap_c1_c2 = 0
        # Create two class for compute overlap between them
        set_coun1 = set(coun_cat_videoid[cat][Coun1])
        set_coun2 = set(coun_cat_videoid[cat][Coun2])

        overlap_c1_c2 = set_coun1 & set_coun2

        c1_num = len(set_coun1)
        c2_num = len(overlap_c1_c2)
        c2_len = len(set_coun1)

        if c2_len != 0:
            c2_prec = c2_num / c2_len
        else:
            c2_prec = 0

        output = "{}; total: {}; {:.1%} in {}".format(
            cat, c1_num, float(c2_prec), Coun2)

        print(output)


if __name__ == "__main__":
    Coun1 = sys.argv[1]
    Coun2 = sys.argv[2]
    reducer(Coun1, Coun2)
