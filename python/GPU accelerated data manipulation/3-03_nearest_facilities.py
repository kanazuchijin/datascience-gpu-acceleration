import cudf
import cuml
import cupy as cp


gdf = cudf.read_csv('./data/week2.csv')

hosp_gdf = cudf.read_csv('./data/hospitals.csv')
#print(hosp_gdf.dtypes)
#print(hosp_gdf.shape)
clinic_gdf = cudf.read_csv('./data/clinics.csv')
#print(clinic_gdf.dtypes)
#print(clinic_gdf.shape)
all_med = cudf.concat([hosp_gdf,clinic_gdf])
#print(all_med.dtypes)
#print(all_med.shape)

# Remove rows that contain null in Latitude and Longitude
all_med2 = all_med.dropna(axis=0, how='any', subset=['Latitude','Longitude'])
all_med2.shape

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