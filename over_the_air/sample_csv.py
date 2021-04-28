# Samples Youtube ID's for a set of labels kinetics 400 csv files
# Creates a new csv with all the URL's that will be downloaded
# csv file contains [URL, label]
# Requires 3 files: train.csv, validate.csv, test.csv from kinetics 400


import pandas as pd
import random
import csv
import argparse
from typing import List


def getSamplesWithLabels(all_samples: pd.DataFrame,
                         class_id: List,
                         skip_ids: List = None):
    skip_ids = skip_ids or []
    out = []
    for _, r in all_samples[all_samples.label.isin(class_id)].iterrows():
        if r['youtube_id'] in skip_ids:
            continue
        out.append(r.values.tolist())

    return out


if __name__ == "__main__":
    seed = 1995
    random.seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', help='Comma Separated List of invalid ids to skip', default='')
    args = parser.parse_args()
    invalid_ids = args.skip.split(',')

    # print(pd.read_csv("test.csv")["label"].unique())
    # all_test_labels = pd.read_csv("test.csv")["label"].unique()

    # Read the 3 csv files    
    train = pd.read_csv("train.csv")
    validate = pd.read_csv("validate.csv")
    test = pd.read_csv("test.csv")

    # Sample 10 labels (hard-coded for now)
    sample_test_labels = ['extinguishing fire', 'bartending', 'ironing',
                          'triple jump', 'playing drums', 'arm wrestling',
                          'planting trees', 'juggling balls', 'shooting goal (soccer)', 'high jump']
    # sample_test_labels = random.sample(set(all_test_labels), 10)
    # print(sample_test_labels)

    # NOTE: In the CSV file, the labels are replaced with numbers from 0-9
    # corresponding to the 10 labels that we chose above.
    # 1: extinguishing fire
    # 2: bartending
    # 3: ironing
    # 4: triple jump
    # 5: playing drums
    # 6: arm wrestling
    # 7: planting trees
    # 8: juggling balls
    # 9: shooting goal (soccer)
    # 10: high jump

    # CSV file name
    filename = 'sampled_urls.csv'

    # Open the CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the CSV field names
        fields = train.columns
        csvwriter.writerow(fields)

        # Filter links that match the label for each file (train, validate, test)
        train_ids = getSamplesWithLabels(train, sample_test_labels, invalid_ids)
        validate_ids = getSamplesWithLabels(validate, sample_test_labels, invalid_ids)
        test_ids = getSamplesWithLabels(test, sample_test_labels, invalid_ids)
        # Choose 4 for train, 2 for validate, 2 for test
        train_sample_ids = random.sample(range(len(train_ids)), 4 * len(sample_test_labels))
        validate_sample_ids = random.sample(range(len(validate_ids)), 2 * len(sample_test_labels))
        test_sample_ids = random.sample(range(len(test_ids)), 2 * len(sample_test_labels))

        # Add data rows to the CSV file for this label
        # Note: this section used to append the label as a string, not a number
        for k in train_sample_ids:
            # csvwriter.writerow([train_sample_ids[k], sample_id])
            csvwriter.writerow(train_ids[k])
        for k in validate_sample_ids:
            # csvwriter.writerow([validate_sample_ids[k], sample_id])
            csvwriter.writerow(validate_ids[k])
        for k in test_sample_ids:
            # csvwriter.writerow([test_sample_ids[k], sample_id])
            csvwriter.writerow(test_ids[k])

        # Let the user know when the program is finished running (since it takes
        # a bit of time to write the csv)
        print("CSV is complete.")
