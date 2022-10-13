import cudf
import cuml
import cupy as cp

gdf = cudf.read_csv('./data/week2.csv')

hosp_gdf = cudf.read_csv('./data/hospitals.csv')
#print(hosp_gdf.dtypes)
#print(hosp_gdf.shape)
print(hosp_gdf[['Latitude','Longitude']].isnull().sum())

clinic_gdf = cudf.read_csv('./data/clinics.csv')
#print(clinic_gdf.dtypes)
#print(clinic_gdf.shape)
print(clinic_gdf[['Latitude','Longitude']].isnull().sum())

all_med = cudf.concat([hosp_gdf[['Latitude','Longitude']],clinic_gdf[['Latitude','Longitude']]], ignore_index=True)
#print(all_med.dtypes)
#print(all_med.shape)
print(all_med.isnull().sum())

# Remove rows that contain null in Latitude and Longitude
all_med2 = all_med.dropna(axis=0, how='any', subset=['Latitude','Longitude'])
#print(all_med2.shape)
#print(all_med2.dtypes)
#print(all_med2.isnull().sum())

# Make Grid Coordinates for Medical Facilities
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

grid_n, grid_e = latlong2osgbgrid_cupy(cp.asarray(all_med2['Latitude']), cp.asarray(all_med2['Longitude']))
#print(len(grid_n))
#print(len(grid_e))

all_med2.reset_index(drop=True, inplace=True)
all_med2['northing'] = cudf.Series(grid_n).astype('float64')
all_med2['easting'] = cudf.Series(grid_e).astype('float64')
#print(all_med2.isnull().sum())

# Find closest hospital or clinic for infected
knn = cuml.NearestNeighbors(n_neighbors=1)
knn.fit(all_med2[['northing','easting']])

# Generate northing and easting values for infected
infected_gdf = gdf[gdf['infected'] == True].reset_index()
grid_n2, grid_e2 = latlong2osgbgrid_cupy(cp.asarray(infected_gdf['lat']), cp.asarray(infected_gdf['long']))

infected_gdf['northing'] = cudf.Series(grid_n2).astype('float32')
infected_gdf['easting'] = cudf.Series(grid_e2).astype('float32')

# Use knn.kneighbors with n_neighbors=1 on infected_gdf's northing and easting values. Save the return values in distances and indices.

distances, indices = knn.kneighbors(infected_gdf[['easting', 'northing']], n_neighbors=1)

# Check Your Solution
# indices, returned from your use of knn.kneighbors immediately above, should map person indices to their closest clinic/hospital indices:

print(indices[:15])
print(indices.shape)

'''
Rows are the infected individuals
0     16696
1       686
2     11757
3     11757
4     16696
5      5573
6     16696
7     11757
8     11757
9       686
10     5091
11    15137
12    12846
13     5091
14     5789
'''

# Here you can print an infected individual's coordinates from infected_gdf;
# get the coords of an infected individual (in this case, individual 0):
infected_gdf.iloc[0]

'''
Results in the following:
index       1.346586e+06
lat         5.371583e+01
long       -2.430079e+00
infected    1.000000e+00
northing    4.244898e+05
easting     3.716197e+05
Name: 0, dtype: float64
'''

# You should be able to used the mapped index for the nearest facility to see that indeed the nearest facility is at a nearby coordinate:
# printing the entry for facility 1234 (replace with the index identified as closest to the individual)
all_med2.iloc[16696]

'''
Latitude         53.246147
Longitude        -1.617808
northing     372224.495976
easting      425500.435427
Name: 16696, dtype: float64
'''