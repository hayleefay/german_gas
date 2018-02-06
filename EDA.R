data <- read.csv("german_gas/gasoline.csv")
names(data)

attach(data)

#intid
hist(data$intid)
stations <- table(data$intid)
station_freq <- as.vector(stations)
summary(station_freq)
length(station_freq[station_freq==1])
length(station_freq[station_freq!=1])
length(station_freq[station_freq==575])
table(station_freq[station_freq!=575])

#marke
length(table(marke))
table(data$marke)[table(data$marke)==max(table(data$marke))]
summary(as.vector(table(marke)))

#autobahn
hist(data$autobahn)
mean(table(data$autobahn))

hist(data$day)
hist(data$rotterdam)
hist(data$vehicles)
hist(data$month)

# create average for price for each month in data set for each brand and check volatility
# over average crude oil
aral <- data[data$aral==1, ]
shell <- data[data$shell==1, ]
esso <- data[data$esso==1, ]
jet <- data[data$jet==1, ]
total <- data[data$total==1, ]

# create list for each brand with average price for each month
months = c(1,2,3,4,5,6,7,8,9,10,11,12)
for (month in months){
  month_df <- shell[data[data$month==month]]
  
}