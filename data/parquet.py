import pandas as pd

# Load parquet
df = pd.read_parquet("options_data_2023.parquet")

# Show 10 rows
print(df.head(10))

# Make sure all columns are shown
pd.set_option("display.max_columns", None)

# (Optional) widen the output so columns don’t get squished
pd.set_option("display.width", None)
