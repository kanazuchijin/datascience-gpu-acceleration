import cudf
import cuml

import cupy as cp

gdf = cudf.read_csv('./data/week1.csv', dtype=['int16', 'str', 'float64', 'float64', 'str', 'boolean'])

print(gdf.columns)
print(gdf.dtypes)

#gdf['infected'].head()

infected_df = gdf[gdf['infected'] == True].reset_index()

print(infected_df[:5])

'''
Provided for you in the next cell (which you can expand by clicking on the "..." and contract again after executing by clicking on the blue left border of the cell) is the lat/long to OSGB36 grid coordinates converter you used earlier in the workshop. Use this converter to create grid coordinate values stored in northing and easting columns of the infected_df you created in the last step.
'''
def latlong2osgbgrid_cupy(lat, long, input_degrees=True):
    '''
    Converts latitude and longitude (ellipsoidal) coordinates into northing and easting (grid) coordinates, using a Transverse Mercator projection.
    
    Inputs:
    lat: latitude coordinate (N)
    long: longitude coordinate (E)
    input_degrees: if True (default), interprets the coordinates as degrees; otherwise, interprets coordinates as radians
    
    Output:
    (northing, easting)
    '''
    
    if input_degrees:
        lat = lat * cp.pi/180
        long = long * cp.pi/180

    a = 6377563.396
    b = 6356256.909
    e2 = (a**2 - b**2) / a**2

    N0 = -100000 # northing of true origin
    E0 = 400000 # easting of true origin
    F0 = .9996012717 # scale factor on central meridian
    phi0 = 49 * cp.pi / 180 # latitude of true origin
    lambda0 = -2 * cp.pi / 180 # longitude of true origin and central meridian
    
    sinlat = cp.sin(lat)
    coslat = cp.cos(lat)
    tanlat = cp.tan(lat)
    
    latdiff = lat-phi0
    longdiff = long-lambda0

    n = (a-b) / (a+b)
    nu = a * F0 * (1 - e2 * sinlat ** 2) ** -.5
    rho = a * F0 * (1 - e2) * (1 - e2 * sinlat ** 2) ** -1.5
    eta2 = nu / rho - 1
    M = b * F0 * ((1 + n + 5/4 * (n**2 + n**3)) * latdiff - 
                  (3*(n+n**2) + 21/8 * n**3) * cp.sin(latdiff) * cp.cos(lat+phi0) +
                  15/8 * (n**2 + n**3) * cp.sin(2*(latdiff)) * cp.cos(2*(lat+phi0)) - 
                  35/24 * n**3 * cp.sin(3*(latdiff)) * cp.cos(3*(lat+phi0)))
    I = M + N0
    II = nu/2 * sinlat * coslat
    III = nu/24 * sinlat * coslat ** 3 * (5 - tanlat ** 2 + 9 * eta2)
    IIIA = nu/720 * sinlat * coslat ** 5 * (61-58 * tanlat**2 + tanlat**4)
    IV = nu * coslat
    V = nu / 6 * coslat**3 * (nu/rho - cp.tan(lat)**2)
    VI = nu / 120 * coslat ** 5 * (5 - 18 * tanlat**2 + tanlat**4 + 14 * eta2 - 58 * tanlat**2 * eta2)

    northing = I + II * longdiff**2 + III * longdiff**4 + IIIA * longdiff**6
    easting = E0 + IV * longdiff + V * longdiff**3 + VI * longdiff**5

    return(northing, easting)

grid_n, grid_e = latlong2osgbgrid_cupy(cp.asarray(infected_df['lat']), cp.asarray(infected_df['long']))

infected_df['northing'] = cudf.Series(grid_n).astype('float32')
infected_df['easting'] = cudf.Series(grid_e).astype('float32')

# find clusters of infected people
'''
Use DBSCAN to find clusters of at least 25 infected people where no member is more than 2000m from at least one other cluster member. Create a new column in infected_df which contains the cluster to which each infected person belongs.
'''
dbscan = cuml.DBSCAN(eps=2000, min_samples=25)
infected_df['cluster'] = dbscan.fit_predict(infected_df[['northing', 'easting']])
infected_df['cluster'].nunique()

# find the centroid of each cluster
print(infected_df.columns)

clusters_north = infected_df[['cluster', 'northing']].groupby(['cluster'])
avg_north = clusters_north.mean()
print(avg_north[:15])

clusters_east = infected_df[['cluster', 'easting']].groupby(['cluster'])
avg_east = clusters_east.mean()
print(avg_east[:15])

# find the number of people in each cluster
infected_value_counts = infected_df['cluster'].value_counts()
print(infected_value_counts.sort_values(ascending=False))

