# german_gas
This is the code for the thesis for my master's degree in computational social science

## TODO

### Data prep
- [x] Add integer of days passed and reorder
- [x] Remove data from final day because no lag available

### Exploratory
- [ ] Create maps (color coded accuracy)
- [ ] Find examples
- [ ] Look at price variation among regions/etc.

### Linear

### Neural Net
- [ ] One hot encode `marke`
- [ ] Try test cases of predictions
- [x] Run on Google Cloud
- [ ] Try without various predictors and see how successful for how many days
- [x] Normalize data
- [ ] Measure diminishing accuracy and plot
- [ ] So if I don't reorder by date before splitting train/test then the error is much lower. The error is lower when the data is ordered by station and then num_days when split rather than num_days and then station. This means it is better at predicting entire stations than it is at predicting that last however many days of ALL stations.
- [ ] Figure out how to build it for more than one time period in the future

### Random forest
- [x] Tune on Google Cloud

### Google Cloud Commands
- SSH: `gcloud compute --project "german-zipcodes-1520272242108" ssh --zone "us-central1-f" "instance-1"`
- Copy (small) files: `gcloud compute scp cloud_train.py "instance-1":~/gas/code/cloud_train.py`
 - Do not do this while SSH'd
- `nohup python3 -u file.py &`
 - The file it outputs to is `nohup.out`
 - `cd` into repo with file
- `ps -e | grep py`
 - To check if running
