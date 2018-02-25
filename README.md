# german_gas
This is the code for the thesis for my master's degree in computational social science

## TODO

### Exploratory
- [ ] Create maps
- [ ] Find examples
- [ ] Try basic regression
- [ ] Look at price variation among regions/etc.

### Neural Net
- [ ] One hot encode `marke`
- [ ] Try test cases of predictions
- [ ] Check interpretation
- [ ] Clean up code
- [ ] Run on Google Cloud
- [ ] Try without various predictors and see how successful for how many days
- [ ] Try mean absolute error
- [ ] Normalize data
- [ ] Create my own training/validation sets so that it is predicting future rather than intermediate days
- [ ] Measure diminishing accuracy and plot

- [ ] How do I get it to just predict the price for the sequence? Or does it need to predict all?

### Random forest with chunking

### Panel OLS/VAR
- [x] Add integer for each station in the data

### Questions for Greg and Amitabh
1. Is there a good way to use a neural network for panel data to predict one of the input variables in a future time period?
2. Should I plan to give it the other inputs in the future dates or not?
3. I am using Keras and an LSTM structure. Does this data format make sense? What about the order?
4. How do latitude and longitude work in this?
5. How to build in lags with panel data? Is it necessary to build lags into the rows to be self-contained?
6. Should I build in the lags myself?
7. With the `lstm.py`, it is killing itself due to I believe lack of memory with my current set up of all stations in a time period belonging in one row. Maybe it would do that anyway. Need more RAM?

### Questions for Dr. Anselin
1. Should I format my data grouped into rows by date?
2. With panel-ols how does longitude and latitude come into this?
