## Introduction to cuDF
'''
You will begin your accelerated data science training with an introduction to cuDF, the RAPIDS API that enables you to create and manipulate GPU-accelerated dataframes. cuDF implements a very similar interface to Pandas so that Python data scientists can use it with very little ramp up. Throughout this notebook we will provide Pandas counterparts to the cuDF operations you perform to build your intuition about how much faster cuDF can be, even for seemingly simple operations.
'''

## Objectives
'''
By the time you complete this notebook you will be able to:
-Read and write data to and from disk with cuDF
-Perform basic data exploration and cleaning operations with cuDF
'''

## Imports
'''
Here we import cuDF and CuPy for GPU-accelerated dataframes and math operations, plus the CPU libraries Pandas and NumPy on which they are based and which we will use for performance comparisons:
'''

import cudf
import cupy as cp

import pandas as pd
import numpy as np

## Reading and Writing Data
'''
Using cuDF, the RAPIDS API providing a GPU-accelerated dataframe, we can read data from a variety of formats, including csv, json, parquet, feather, orc, and Pandas dataframes, among others.

For the first part of this workshop, we will be reading almost 60 million records (corresponding to the entire population of England and Wales) which were sythesized from official UK census data. Here we read this data from a local csv file directly into GPU memory:
'''

%time gdf = cudf.read_csv('./data/pop_1-03.csv')
gdf.shape
gdf.dtypes

'''
Here for comparison we read the same data into a Pandas dataframe:
'''

%time df = pd.read_csv('./data/pop_1-03.csv')
gdf.shape == df.shape

'''
Because of the sophisticated GPU memory management behind the scenes in cuDF, the first data load into a fresh RAPIDS memory environment is sometimes substantially slower than subsequent loads. The RAPIDS Memory Manager is preparing additional memory to accommodate the array of data science operations that you may be interested in using on the data, rather than allocating and deallocating the memory repeatedly throughout your workflow.

We will be using gdf regularly in this workshop to represent a GPU dataframe, as well as df for a CPU dataframe when comparing performance.
'''

## Writing to File
'''
cuDF also provides methods for writing data to files. Here we create a new dataframe specifically containing residents of Blackpool county and then write it to blackpool.csv, before doing the same with Pandas for comparison.
'''
# cuDF
%time blackpool_residents = gdf.loc[gdf['county'] == 'BLACKPOOL']
print(f'{blackpool_residents.shape[0]} residents')
%time blackpool_residents.to_csv('blackpool.csv')

#Pandas
%time blackpool_residents_pd = df.loc[df['county'] == 'BLACKPOOL']
%time blackpool_residents_pd.to_csv('blackpool_pd.csv')

## Exercise: Initial Data Exploration
'''
Now that we have some data loaded, let's do some initial exploration.

Use the head, dtypes, and columns methods on gdf, as well as the value_counts on individual gdf columns, to orient yourself to the data. If you're interested, use the %time magic command to compare performance against the same operations on the Pandas df.

You can create additional interactive cells by clicking the + button above, or by switching to command mode with Esc and using the keyboard shortcuts a (for new cell above) and b (for new cell below).

If you fill up the GPU memory at any time, don't forget that you can restart the kernel and rerun the cells up to this point quite quickly.
'''

gdf.head()
gdf.columns
gdf.dtypes
gdf['age'].value_counts

## Basic Operations with cuDF
'''
Except for being much more performant with large datasets, cuDF looks and feels a lot like Pandas. In this section we highlight a few very simple operations. When performing data operations on cuDF dataframes, column operations are typically much more performant than row-wise operations.
'''

## Converting Data Types
'''
For machine learning later in this workshop, we will sometimes need to convert integer values into floats. Here we convert the age column from int64 to float32, comparing performance with Pandas:
'''
# cuDF
%time gdf['age'] = gdf['age'].astype('float32')

# Pandas
%time df['age'] = df['age'].astype('float32')

## Column-Wise Aggregations
'''
Similarly, column-wise aggregations take advantage of the GPU's architecture and RAPIDS' memory format.
'''
# cuDF
%time gdf['age'].mean()

# Pandas
%time df['age'].mean()

## String Operations
'''
Although strings are not a datatype traditionally associated with GPUs, cuDF supports powerful accelerated string operations.
'''
# cuDF
%time gdf['name'] = gdf['name'].str.title()
gdf.head()

# Pandas
%time df['name'] = df['name'].str.title()
df.head()

## Data Subsetting with loc and iloc
'''
cuDF also supports the core data subsetting tools loc (label-based locator) and iloc (integer-based locator).
'''

# Range Selection
'''
Our data's labels happen to be incrementing numbers, though as with Pandas, loc will include every value it is passed whereas iloc will give the half-open range (omitting the final value).
'''

gdf.loc[100:105]
gdf.iloc[100:105]

# loc with Boolean Selection
'''
We can use loc with boolean selections:
'''
# cuDF
%time e_names = gdf.loc[gdf['name'].str.startswith('E')]
e_names.head()

# Pandas
%time e_names_pd = df.loc[df['name'].str.startswith('E')]
e_names_pd.head()

## Combining with NumPy Methods
'''
We can combine cuDF methods with NumPy methods, just like Pandas. Here we use np.logical_and for elementwise boolean selection.
'''
# cuDF
%time ed_names = gdf.loc[np.logical_and(gdf['name'].str.startswith('E'), gdf['name'].str.endswith('d'))]
ed_names.head()

'''
For better performance at scale, we can use CuPy instead of NumPy, thereby performing the elementwise boolean logical_and operation on GPU.
'''

%time ed_names = gdf.loc[cp.logical_and(gdf['name'].str.startswith('E'), gdf['name'].str.endswith('d'))]
ed_names.head()

# Pandas
%time ed_names_pd = df.loc[np.logical_and(df['name'].str.startswith('E'), df['name'].str.endswith('d'))]

## Exercise: Basic Data Cleaning
'''
For this exercise we ask you to perform two simple data cleaning tasks using several of the techniques described above:
- Modifying the data type of a couple columns
- Transforming string data into our desired format
'''
# 1. Modify dtypes
'''
Examine the dtypes of gdf and convert any 64-bit data types to their 32-bit counterparts.
'''

gdf['lat'] = gdf['lat'].astype('float32')
gdf['long'] = gdf['long'].astype('float32')

# 2. Title Case the Counties
'''
As it stands, all of the counties are UPPERCASE:
'''

gdf['county'].head()

# Convert to title case:
gdf['county'] = gdf['county'].str.title()

## Exercise: Counties North of Sunderland
'''
This exercise will require to use the loc method, and several of the techniques described above. Identify the latitude of the northernmost resident of Sunderland county (the person with the maximum lat value), and then determine which counties have any residents north of this resident. Use the unique method of a cudf Series to deduplicate the result.
'''

sunderland_residents = gdf.loc[gdf['county'] == 'Sunderland']
northmost_sunderland_lat = sunderland_residents['lat'].max()
counties_with_pop_north_of = gdf.loc[gdf['lat'] > northmost_sunderland_lat]['county'].unique()