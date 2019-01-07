
# https://www.pypandas.cn/document/10min.html
# 10分钟入门Pandas

# 目录
# Object Creation
# Viewing Data
# Selection
# Getting
# Selection by Label
# Selection by Position
# Boolean Indexing
# Setting
# Missing Data
# Operations
# Stats
# Apply
# Histogramming
# String Methods
# Merge
# Concat
# Join
# Append
# Grouping
# Reshaping
# Stack
# Pivot Tables
# Time Series
# Categoricals
# Plotting
# Getting Data In/Out
# CSV
# HDF5
# Excel
# Gotchas



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

dates = pd.date_range('20160101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
print(df2)
print(df2.dtypes)

print(df.head(n=3))
print(df.tail(3))
print(df.describe())