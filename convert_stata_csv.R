library(haven)
yourData = read_dta("Fulldata.dta")
write.csv(yourData, file = "gasonline.csv")
