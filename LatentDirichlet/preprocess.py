import collections
import csv
import simplejson as json
import numpy as np
import pandas as pd


def read_and_write_file(json_file_path, csv_file_path, column_names):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    with open(csv_file_path, 'wb+') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(json_file_path) as fin:
            for line in fin:
                line_contents = json.loads(line)
                csv_file.writerow(get_row(line_contents, column_names))


def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""
    column_names = set()
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                set(get_column_names(line_contents).keys())
            )
    return column_names


def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.
    Example:
        line_contents = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        will return: ['a.b', 'a.c']
    These will be the column names for the eventual csv file.
    """
    column_names = []
    for k, v in line_contents.iteritems():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                get_column_names(v, column_name).items()
            )
        else:
            column_names.append((column_name, v))
    return dict(column_names)


def get_nested_value(d, key):
    if '.' not in key:
        if key not in d:
            return None
        return d[key]
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    return get_nested_value(sub_dict, sub_key)


def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        line_value = get_nested_value(
            line_contents,
            column_name,
        )
        if isinstance(line_value, unicode):
            row.append('{0}'.format(line_value.encode('utf-8')))
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row

    """Convert a yelp dataset file from json to csv."""



if __name__ == '__main__':


    json_file = "./yelp_dataset_challenge_round9/yelp_academic_dataset_business.json"
    csv_file = '{0}.csv'.format(json_file.split('.json')[0])
    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)
    """Convert a yelp dataset file from json to csv."""

    json_file = "./yelp_dataset_challenge_round9/yelp_academic_dataset_review.json"
    csv_file = '{0}.csv'.format(json_file.split('.json')[0])
    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)


    # separate businesses from different cities according to states to different csv
    data = pd.read_csv("./yelp_dataset_challenge_round9/yelp_academic_dataset_business.csv")
    print list(data)
    for state in ['AZ','NV','OH','PA','WI','IL','NC']:
        csv_file = "./data/"+str(state)+"business.csv"
        #new = data[data['state'] == state]
        new = data[(data['state']==state) & (data['categories'].str.contains("Restaurants"))]
        new.to_csv(csv_file)




    # separate reviews from different cities according to states to different csv
    # only for business with greater than 10 reviews
    data = pd.read_csv("./yelp_dataset_challenge_round9/yelp_academic_dataset_review.csv")
    for state in ['AZ', 'NV', 'OH', 'PA', 'WI', 'IL', 'NC']:
        bus = pd.read_csv("./data/" + str(state) + "business.csv")
        unique_bus_id = bus.business_id.unique()
        review = data[data['business_id'].isin(unique_bus_id)]
        filtered_id = review['business_id'].value_counts()
        filtered = filtered_id[filtered_id >= 100].index.tolist()
        review = data[data['business_id'].isin(filtered)]
        review.to_csv("./data/" + str(state) + "review.csv")










