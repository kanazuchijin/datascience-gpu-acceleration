# Grid Coordinate Conversion with CuPy
'''
Much of our data is provided with latitude and longitude coordinates, but for some of our machine learning tasks involving distance - identifying geographically dense clusters of infected people, locating the nearest hospital or clinic from a given person - it is convenient to have Cartesian grid coordinates instead. Our road data comes with those coordinates, as well. By using a region-specific map projection - in this case, the Ordnance Survey Great Britain 1936 - we can compute local distances efficiently and with good accuracy.

In this notebook you will use a user-defined function to perform data manipulation, generating grid coordinate values. In doing so, you will learn more about the powerful GPU-accelerated drop-in replacement library for NumPy called CuPy.
'''

# Objectives
'''
By the time you complete this notebook you will be able to:
- Use CuPy to GPU-accelerate data transformations using user-defined functions
'''

## Imports
import cudf

import numpy as np
import cupy as cp

# Read Data
'''
For this notebook we will load the UK population data again. Here and later, when reading data from disk we will provide the named argument dtype to specify the data types we want the columns to load as.
'''

%time gdf = cudf.read_csv('./data/pop_1-05.csv', dtype=['float32', 'str', 'str', 'float32', 'float32', 'str'])
gdf.dtypes
gdf.shape

# Lat/Long to OSGB Grid Converter with NumPy
'''
To perform coordinate conversion, we will create a function latlong2osgbgrid which accepts latitude/longitude coordinates and converts them to OSGB36 coordinates: "northing" and "easting" values representing the point's Cartesian coordinate distances from the southwest corner of the grid. See https://en.wikipedia.org/wiki/Ordnance_Survey_National_Grid.

Immediately below is latlong2osgbgrid, which relies heavily on NumPy:
'''

# https://www.ordnancesurvey.co.uk/docs/support/guide-coordinate-systems-great-britain.pdf

def latlong2osgbgrid(lat, long, input_degrees=True):
    '''
    Converts latitude and longitude (ellipsoidal) coordinates into northing and easting (grid) coordinates, using a Transverse Mercator projection.
    
    Inputs:
    lat: latitude coordinate (north)
    long: longitude coordinate (east)
    input_degrees: if True (default), interprets the coordinates as degrees; otherwise, interprets coordinates as radians
    
    Output:
    (northing, easting)
    '''
    
    if input_degrees:
        lat = lat * np.pi/180
        long = long * np.pi/180

    a = 6377563.396
    b = 6356256.909
    e2 = (a**2 - b**2) / a**2

    N0 = -100000                # northing of true origin
    E0 = 400000                 # easting of true origin
    F0 = .9996012717            # scale factor on central meridian
    phi0 = 49 * np.pi / 180     # latitude of true origin
    lambda0 = -2 * np.pi / 180  # longitude of true origin and central meridian
    
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    tanlat = np.tan(lat)
    
    latdiff = lat-phi0
    longdiff = long-lambda0

    n = (a-b) / (a+b)
    nu = a * F0 * (1 - e2 * sinlat ** 2) ** -.5
    rho = a * F0 * (1 - e2) * (1 - e2 * sinlat ** 2) ** -1.5
    eta2 = nu / rho - 1
    M = b * F0 * ((1 + n + 5/4 * (n**2 + n**3)) * latdiff - 
                  (3*(n+n**2) + 21/8 * n**3) * np.sin(latdiff) * np.cos(lat+phi0) +
                  15/8 * (n**2 + n**3) * np.sin(2*(latdiff)) * np.cos(2*(lat+phi0)) - 
                  35/24 * n**3 * np.sin(3*(latdiff)) * np.cos(3*(lat+phi0)))
    I = M + N0
    II = nu/2 * sinlat * coslat
    III = nu/24 * sinlat * coslat ** 3 * (5 - tanlat ** 2 + 9 * eta2)
    IIIA = nu/720 * sinlat * coslat ** 5 * (61-58 * tanlat**2 + tanlat**4)
    IV = nu * coslat
    V = nu / 6 * coslat**3 * (nu/rho - np.tan(lat)**2)
    VI = nu / 120 * coslat ** 5 * (5 - 18 * tanlat**2 + tanlat**4 + 14 * eta2 - 58 * tanlat**2 * eta2)

    northing = I + II * longdiff**2 + III * longdiff**4 + IIIA * longdiff**6
    easting = E0 + IV * longdiff + V * longdiff**3 + VI * longdiff**5

    return(northing, easting)

# Testing the NumPy Converter
'''
To test the converter and check its performance, here we generate 10,000,000 normally distributed random coordinates within the rough bounds of the latitude and longitude ranges of the UK.
'''

%%time
coord_lat = np.random.normal(54, 1, 10000000)
coord_long = np.random.normal(-1.5, .25, 10000000)

'''
We now pass these latitude/longitude coordinates into the converter, which returns north and east values within the OSGB grid:
'''

%time grid_n, grid_e = latlong2osgbgrid(coord_lat, coord_long)
print(grid_n[:5], grid_e[:5])

# Lat/Long to OSGB Grid Converter with CuPy
'''
CuPy is a NumPy-like matrix library that can often be used as a drop in replacement for NumPy.

In the following latlong2osgbgrid_cupy, we simply swap cp in for np. While CuPy supports a wide variety of powerful GPU-accelerated tasks, this simple technique of being able to swap in CuPy calls for NumPy calls makes it an incredibly powerful tool to have at your disposal.
'''

def latlong2osgbgrid_cupy(lat, long, input_degrees=True):
    '''
    Converts latitude and longitude (ellipsoidal) coordinates into northing and easting (grid) coordinates, using a Transverse Mercator projection.
    
    Inputs:
    lat: latitude coordinate (north)
    long: longitude coordinate (east)
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

    N0 = -100000                 # northing of true origin
    E0 = 400000                  # easting of true origin
    F0 = .9996012717             # scale factor on central meridian
    phi0 = 49 * cp.pi / 180      # latitude of true origin
    lambda0 = -2 * cp.pi / 180   # longitude of true origin and central meridian
    
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

# Testing the CuPy Converter
'''
Here we perform the same operations as we did with NumPy above, only the conversion runs significantly faster. Once you have run the cells below, try rerunning the NumPy converter above (including random number generation) and then the CuPy converter - you may see even larger differences.
'''

coord_lat = cp.random.normal(54, 1, 10000000)
coord_long = cp.random.normal(-1.5, .25, 10000000)

grid_n, grid_e = latlong2osgbgrid_cupy(coord_lat, coord_long)
print(grid_n[:5], grid_e[:5])

## Adding Grid Coordinate Columns to Dataframe
'''
Now we will utilize latlong2osgbgrid_cupy to add northing and easting columns to gdf. We start by converting the two columns we need, lat and long, to CuPy arrays with the cp.asarray method. Because cuDF and CuPy interface directly via the __cuda_array_interface__, the conversion can happen in nanoseconds.
'''

cupy_lat = cp.asarray(gdf['lat'])
cupy_long = cp.asarray(gdf['long'])

## Exercise: Create Grid Columns
'''
For this exercise, now that you have GPU arrays for lat and long, you will create northing and easting columns in gdf. To do this:
- Use latlong2osgbgrid_cupy with cupy_lat and cupy_long, just created, to make CuPy arrays of the grid coordinates
- Create cuDF series out of each of these coordinate CuPy arrays and set the dtype to float32
- Add these two new series to gdf, calling them northing and easting
'''

n_cupy_array, e_cupy_array = latlong2osgbgrid_cupy(cupy_lat, cupy_long)
gdf['northing'] = cudf.Series(n_cupy_array).astype('float32')
gdf['easting'] = cudf.Series(e_cupy_array).astype('float32')
print(gdf.dtypes)
gdf.head()
