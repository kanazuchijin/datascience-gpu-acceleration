import numpy as np # for generating sample data

import pandas as df
import time # for clocking process times
import matplotlib.pyplot as plt # for visualizing results

class Timer: # creating a Timer helper class to measure execution time
    def __enter__(self):
        self.start=time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end=time.perf_counter()
        self.interval=self.end-self.start

#=======================================#
# Load the data into two data frames    #
#=======================================#
rows=1000000
columns=50

def load_data(): 
    data_a=np.random.randint(0, 100, (rows, columns))
    data_b=np.random.randint(0, 100, (rows, columns))
    dataframe_a=df.DataFrame(data_a, columns=[f'a_{i}' for i in range(columns)])
    dataframe_b=df.DataFrame(data_b, columns=[f'b_{i}' for i in range(columns)])
    return dataframe_a, dataframe_b

with Timer() as process_time: 
    dataframe_a, dataframe_b=load_data()
    print(f'The loading process took {process_time.interval:.2f} seconds')

#=======================================#
# Merge the data frames into just one   #
#=======================================#
dataframe_a.head(5)
dataframe_b.head(5)

def merge_data(left_df, right_df):
    combined_df=df.merge(left_df, right_df, left_index=True, right_index=True)
    return combined_df

with Timer() as process_time: 
    combined_df=merge_data(dataframe_a, dataframe_b)
    print(f'The merging process took {process_time.interval:.2f} seconds')

combined_df.head()

#=======================================#
# Summarize the data frame              #
#=======================================#
def summarize(dataframe):
    summary_df=dataframe.describe()
    return summary_df

with Timer() as process_time: 
    summary_df=summarize(combined_df)
    print(f'The summarizing process took {process_time.interval:.2f} seconds')

summary_df

#=======================================#
# Perform a correlation analysis        #
#=======================================#
def correlation(dataframe): 
    corr_df=dataframe.corr()
    return corr_df

with Timer() as process_time: 
    corr_df=correlation(combined_df)
    print(f'The correlation process took {process_time.interval:.2f} seconds')

corr_df.head()

#=======================================#
# Group the data using groupby()        #
#=======================================#
def groupby_summarize(dataframe):
    dataframe['group']=dataframe.index
    dataframe['group']=df.cut(dataframe['group'], 5)
    group_describe_df=dataframe.groupby('group').mean().reset_index(drop=True)
    return group_describe_df

with Timer() as process_time: 
    group_describe_df=groupby_summarize(combined_df)
    print(f'The grouping process took {process_time.interval:.2f} seconds')

group_describe_df

#=======================================#
# Time the entire workflow above        #
#=======================================#
def pipeline(df_lib):
    if(df_lib=='cudf'):
        import cudf as df
    elif(df_lib=='pandas'):
        import pandas as df
    performance={}
    with Timer() as process_time: 
        dataframe_a, dataframe_b=load_data()
    performance['load data']=process_time.interval
    with Timer() as process_time: 
        combined_df=merge_data(dataframe_a, dataframe_b)
    performance['merge data']=process_time.interval
    with Timer() as process_time: 
        summarize(combined_df)
    performance['summarize']=process_time.interval
    with Timer() as process_time: 
        correlation(combined_df)
    performance['correlation']=process_time.interval
    with Timer() as process_time: 
        groupby_summarize(combined_df)
    performance['groupby & summarize']=process_time.interval
    # If the data frame library is cuDF, then send to Pandas to plot
    # Else, if it is not, then just plot it using the existing data frame
    if df.__name__=='cudf': 
        df.DataFrame([performance], index=['gpu']).to_pandas().plot(kind='bar', stacked=True)
    else: 
        df.DataFrame([performance], index=['cpu']).plot(kind='bar', stacked=True)
    plt.show()
    return None

# Run the pipeline of work using CPU
# This uses Pandas as the underlying
# data frame architecture
pipeline('pandas')

# Using the cuDF library to generate
# the data frame, the GPU is used
# to process the pipeline
pipeline('cudf')