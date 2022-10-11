## Grouping and Sorting with cuDF
'''
In this notebook you will be introduced to grouping and sorting with cuDF, with performance comparisons to Pandas, before integrating what you learned in a short data analysis exercise.
'''
# Objectives
'''
By the time you complete this notebook you will be able to:
- Perform GPU-accelerated group and sort operations with cuDF
'''

# Imports
import cudf
import pandas as pd

# Read Data
'''
We once again read the UK population data, returning to timed comparisons with Pandas.
'''

%time gdf = cudf.read_csv('./data/pop_1-04.csv', dtype=['float32', 'str', 'str', 'float32', 'float32', 'str'])
%time df = pd.read_csv('./data/pop_1-04.csv')
gdf.dtypes
gdf.shape
gdf.head()

# Grouping and Sorting
# Record Grouping
'''
Record grouping with cuDF works the same way as in Pandas.
'''
# cuDF
%%time
counties = gdf[['county', 'age']].groupby(['county'])
avg_ages = counties.mean()
print(avg_ages[:5])

# Pandas
%%time
counties_pd = df[['county', 'age']].groupby(['county'])
avg_ages_pd = counties_pd.mean()
print(avg_ages_pd[:5])

# Sorting
'''
Sorting is also very similar to Pandas, though cuDF does not support in-place sorting.
'''
# cuDF
%time gdf_names = gdf['name'].sort_values()
print(gdf_names[:5]) # yes, "A" is an infrequent but correct given name in the UK, according to census data
print(gdf_names[-5:])

# Pandas
'''
This operation takes a while with Pandas. Feel free to start the next exercise while you wait.
'''

%time df_names = df['name'].sort_values()
print(df_names[:5])
print(df_names[-5:])

## Exercise: Youngest Names
'''
For this exercise you will need to use both groupby and sort_values.

We would like to know which names are associated with the lowest average age and how many people have those names. Using the mean and count methods on the data grouped by name, identify the three names with the lowest mean age and their counts.
'''

name_groups = gdf[['name', 'age']].groupby('name')

name_ages = name_groups['age'].mean()
name_counts = name_groups['age'].count()

ages_counts = cudf.DataFrame()
ages_counts['mean_age'] = name_ages
ages_counts['count'] = name_counts

ages_counts = ages_counts.sort_values('mean_age')
ages_counts.iloc[:3]

