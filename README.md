# german_gas
This is the code for the thesis for my master's degree in computational social science

## TODO

### Data prep
- [x] Add integer of days passed and reorder
- [ ] Make trend stationary
- [x] Remove data from final day because no lag available

### Exploratory
- [ ] Create maps (color coded accuracy)
- [ ] Find examples
- [ ] Look at price variation among regions/etc.

### Linear

### VAR/Panel OLS
- [x] Add integer for each station in the data
- [ ] Check out plm in R
- [ ] Cross-validation iterators for grouped data

### Neural Net
- [ ] One hot encode `marke`
- [ ] Try test cases of predictions
- [ ] Run on Google Cloud
- [ ] Try without various predictors and see how successful for how many days
- [x] Normalize data
- [ ] Measure diminishing accuracy and plot
- [ ] So if I don't reorder by date before splitting train/test then the error is much lower. The error is lower when the data is ordered by station and then num_days when split rather than num_days and then station. This means it is better at predicting entire stations than it is at predicting that last however many days of ALL stations.

### Random forest

### Questions for Dr. Anselin
1. VAR
2. Remove time trend -- is it actually seasonal and how to without removing first year?
