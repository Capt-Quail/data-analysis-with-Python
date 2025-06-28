from pathlib import Path
import pandas as pd
import numpy as np

student_data = (
    Path(__file__).parent.parent
    /"Data"
    /"student_habits_performance.csv"
)

df = pd.read_csv(student_data)

# Quick preview - understand the data at a glance
# print(df.head(20))

# View structure as a whole
# print(df.info())

# Descriptive stats about the student data
# print(df.describe())

# Checking all of our data types
# print(df.dtypes)

# Probably lastly, viewing if data is missing.
# This is generated data, so I expect nothing is missing. (wrong)
# print(df.isnull())

# A more accurate view of missing values since nothing appeared missing.
missing_values = df.isnull()

for column in missing_values.columns.values.tolist():
    print(missing_values[column].value_counts())
# Interestingly enough there are 91 missing values in parental education.

# Here we use a new method that selects a specific dtype in the df.
# We want to see what exactly is inside the 'object' type columns.
for col in df.select_dtypes(include="object"):
    print(f"Unique values in '{col}':")
    print(df[col].unique())
    print("-" * 40)
# df.replace()
