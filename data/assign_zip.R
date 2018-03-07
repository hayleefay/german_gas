library(raster)
# library(maptools)
library(spatialEco)

# read in the latlon
ll = read.csv('latlon.csv')
ll = within(ll, rm('X'))

coordinates(ll) <- cbind(ll$longitude , ll$latitude)

# proj4string(ll) = CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")

# read in the shapefile
shp <- shapefile('german_zips/plz-gebiete.shp')

# make sure the two files share the same CRS
ll@proj4string <- shp@proj4string

point.in.poly(ll, shp)

df <- over(ll, shp)
