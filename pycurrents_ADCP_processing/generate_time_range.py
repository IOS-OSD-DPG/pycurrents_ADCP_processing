"""
author: Hana Hourston

Python script for generating a time range for ADCP files with invalid time data and exporting the new
time range data in a csv file.
"""

import pandas as pd
import csv


raw_file = './sample_data/eh2_20060530_20060717_0007m.000'
raw_freq = '60Min'

time_file = 'eh2_2006_time.csv'

rng = pd.date_range(start='2005-05-30 06:00:00', periods=1174, freq=raw_freq)

len(rng)

with open(time_file, mode='w', newline='') as out_file:
    csv_writer = csv.writer(out_file, delimiter=',')
    csv_writer.writerow(['time'])
    for timestamp in rng:
        csv_writer.writerow([timestamp])