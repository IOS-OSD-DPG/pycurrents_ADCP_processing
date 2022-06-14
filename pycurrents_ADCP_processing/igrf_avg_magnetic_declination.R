library(oce)

lat = 53+56.266/60
lon = -(128+41.2386/60)
st = '2020-08-02 00:00:00 UTC'  # Start time
et = '2021-06-07 00:00:00 UTC'  # End time

# Calculate average time in a time series
# calculate middle date of timeseries in seconds since origin
m1 <- as.integer(as.POSIXct(st, tz = 'UTC')) + (as.integer(as.POSIXct(et, tz = 'UTC')) - as.integer(as.POSIXct(st, tz = 'UTC')))/2
# Convert integer type to a more readable format
m <- as.POSIXct(m1, tz = 'UTC', origin = '1970-01-01 00:00:00')

# Calculate average magnetic declination over deployment
# R indexes from 1 not 0
# magneticField returns a list containing [mag_decl, mag_incl, mag_intensity]
mag_decl = magneticField(lon, lat, m)[1]

print(mag_decl)