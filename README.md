# german_gas
This is the code for the thesis for my master's degree in computational social science

### Google Cloud Commands
- SSH: `gcloud compute --project "german-zipcodes-1520272242108" ssh --zone "us-central1-f" "instance-1"`
- Copy (small) files: `gcloud compute scp changeslstm_cloud.py "instance-1":~/gas/neural-net/changeslstm_cloud.py`
 - Do not do this while SSH'd
- `nohup python3 -u file.py &`
 - The file it outputs to is `nohup.out`
 - `cd` into repo with file
- `ps -e | grep py`
 - To check if running
- `gsutil cp rf_predictions.csv gs://german-data`
 - to copy files from instance to storage
