from pandas import read_csv
from datetime import datetime


def parse(x):
    return datetime.strptime(x, '%d%b%Y').strftime('%Y %m %d')


dataset = read_csv('gasoline.csv', parse_dates=['date'],
                   index_col=0, date_parser=parse)

# one hot ecode marke


dataset.drop(['mts_id', 'intid', 'year', 'month', 'day', 'vehicles1',
              'latitudezst', 'longitudezst', 'brentl', 'd1', 'zst1'],
             axis=1, inplace=True)
# set the date as the index
# dataset.set_index('date', inplace=True)
# mark all NAs as 0 for now
dataset['e5gas'].fillna(0, inplace=True)
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('prepared_gasoline.csv')
