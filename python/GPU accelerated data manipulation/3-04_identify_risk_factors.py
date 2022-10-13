import cudf
import cuml

gdf = cudf.read_csv('./data/week3.csv', dtype=['int64', 'str', 'str', 'float32'])
#print(gdf.shape)
#print(gdf.dtypes)
#print(gdf[:16])

'''
Now, produce a list of employment types and their associated rates of infection, sorted from highest to lowest rate of infection.

NOTE: The infection rate for each employment type should be the percentage of total individuals within an employment type who are infected. Therefore, if employment type "X" has 1000 people, and 10 of them are infected, the infection rate would be .01. If employment type "Z" has 10,000 people, and 50 of them are infected, the infection rate would be .005, and would be lower than for type "X", even though more people within that employment type were infected.
'''
infected_gdf = gdf[gdf['infected'] == 1.0].reset_index()
employ_infect = cudf.DataFrame(infected_gdf.groupby('employment')['infected'].count(), index=None)
employ_infect.columns = ['infected_count']
#print(employ_infect)

notinfected_gdf = gdf[gdf['infected'] == 0.0].reset_index()
employ_notinfect = cudf.DataFrame(notinfected_gdf.groupby('employment')['infected'].count(), index=None)
employ_notinfect.columns = ['notinfected_count']
#print(employ_notinfect)

employment_type = gdf['employment'].unique()
employment_type.columns = ['employment']
#print(employment_type)

infection_rates = cudf.concat([employment_type, employ_infect, employ_notinfect], axis=1)
#print(infection_rates)

infection_rates2 = infection_rates.assign(total_count = infection_rates['infected_count'] + infection_rates['notinfected_count'])

infection_rates3 = infection_rates2.assign(infect_rate = infection_rates2['infected_count'] / infection_rates2['total_count'])

(infection_rates3
 .sort_values(by=('infect_rate'), ascending=False)
)

# Finally, read in the employment codes guide from ./data/code_guide.csv to interpret which employment types are seeing the highest rates of infection.
employ_codes_df = cudf.read_csv('./data/code_guide.csv')
print(employ_codes_df.shape)
print(employ_codes_df.dtypes)
print(employ_codes_df)

# Calculate Infection Rates by Employment Code and Sex
# We want to see if there is an effect of sex on infection rate, either in addition to employment or confounding it. Group by both employment and sex simultaneously to get the infection rate for the intersection of those categories.

