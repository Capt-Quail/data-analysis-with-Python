from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

filename = Path(__file__).parent.parent/"Data"/"auto.csv"

headers = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight",
    "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
    "bore", "stroke", "compression-ratio", "horsepower",
    "peak-rpm", "city-mpg", "highway-mpg", "price"
]
df = pd.read_csv(filename, names=headers)
# To see the first five rows of the dataset
#print(df.head(5))

# Replace "?" with non-missing values
df.replace("?", np.nan, inplace=True)
#print(df.head(5))

# Create a data structure of the columns missing values using boolean value "True"
missing_data = df.isnull()
#print(missing_data.head(5))

# for column in missing_data.columns.values.tolist():
#     print(missing_data[column].value_counts())
#     print("")

# normalized-losses: 41 missing values - replace with mean
# num-of-doors: 2 missing values - replace with "four"
# bore: 4 missing values - replace with mean
# stroke: 4 missing values = replace with mean
# horsepower: 2 missing values - replace them with mean
# peak-rpm: 2 missing values - replace with mean
# price: 4 missing values - omit entire row

# Fixing missing numerical values
avg_norm_loss = df["normalized-losses"].astype(float).mean()
#print("Average Normalized Loss: ", avg_norm_loss, "\n")
df["normalized-losses"] = df["normalized-losses"].replace(np.nan, avg_norm_loss)

avg_bore = df["bore"].astype(float).mean()
df["bore"] = df["bore"].replace(np.nan, avg_bore)

avg_stroke = df["stroke"].astype(float).mean()
df["stroke"] = df["stroke"].replace(np.nan, avg_stroke)

avg_horses = df["horsepower"].astype(float).mean()
df["horsepower"] = df["horsepower"].replace(np.nan, avg_horses)

avg_rpm = df["peak-rpm"].astype(float).mean()
df["peak-rpm"] = df["peak-rpm"].replace(np.nan, avg_rpm)

# Showing the most common/frequent value in this categorical data
# print(df["num-of-doors"].value_counts().idxmax())
df["bore"] = df["bore"].replace(np.nan, "four")

# Dropna to remove the cars with missing values under our target variable
df = df.dropna(subset="price", axis=0)
# Reset the index since we removed rows with missing values
df = df.reset_index(drop=True)

# Part of the data preparation process is ensuring all data is the correct type
# Confirm with .dtypes
# print(df.dtypes)

# Finish normalization by converting those mismatched types
df[["bore", "stroke", "peak-rpm", "price"]] = df[["bore", "stroke", "peak-rpm", "price"]].astype(float)
# print(df.dtypes)
df["normalized-losses"] = df["normalized-losses"].astype(int)

# We are going to standardize mpg to L/100km to align with a different standard
# You can also create a new column by declaring it
df["city-mpg"] = 235/df["city-mpg"]
df = df.rename(columns={"city-mpg": "city-L/100km"})

# Transform mpg to L/100km and rename ghigh-way mpg to highway-L/100km
df["highway-mpg"] = 235/df["highway-mpg"]
df = df.rename(columns={"highway-mpg": "highway-L/100km"})
# print(df.head())

# Normalization is the process of transforming values of several vars into a similar range
# We will use simple feature scaling to normalize length, width, and height
df["length"] = df["length"]/df["length"].max()
df["width"] = df["width"]/df["width"].max()
df["height"] = df["height"]/df["height"].max()
# print(df[["length", "width", "height"]])

# Convert "horsepower" to correct datatype
df["horsepower"] = df["horsepower"].astype(int)

# Visualizing data
plt.hist(df["horsepower"])
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
# plt.show()

# Divide horsepower into 4 equidistant buckets based on its range as an array
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ["Low", "Medium", "High"]

# df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)
# Show the new columns value next to relevant data
# print(df[["horsepower", "horsepower-binned"]].head(20))
# print(df["horsepower-binned"].value_counts())

# Plotting the distribution of each bin with a bar gra
# plt.bar(group_names, df["horsepower-binned"].value_counts())

# plt.xlabel("horsepower")
# plt.ylabel("count")
# plt.title("horsepower bins")
# plt.show()

# Alternate binning with a histograph (used when you don't want to retain bin data)
# plt.hist(df["horsepower"], bins=3)

# plt.xlabel("horsepower")
# plt.ylabel("count")
# plt.title("horsepower bins")
# plt.show()

# Create dummy variables so our categorical data can be used in modeling
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())
dummy_variable_1 = dummy_variable_1.rename(
    columns={"diesel":"fuel-type-diesel", "gas":"fuel-type-gas"}
)
# print(dummy_variable_1.head())
df = pd.concat([df, dummy_variable_1], axis=1)
df = df.drop("fuel-type", axis=1)

dummy_variable_2 = pd.get_dummies(df["aspiration"])
# print(dummy_variable_2.head())
dummy_variable_2 = dummy_variable_2.rename(
    columns={"std":"aspiration-std", "turbo":"aspiration-turbo"}
)
df = pd.concat([df, dummy_variable_2], axis=1)
df = df.drop("aspiration", axis=1)
# print(df.head())

clean_data = (
    Path(__file__).parent.parent
    /"Data"
    /"Clean_Data"
    /"clean_auto_df.csv"
)
df.to_csv(clean_data, index=False)
