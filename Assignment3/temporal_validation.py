# start time of our data
start_time = '2011-01-01'

#last date of data including labels and outcomes that we have
end_time = '2013-12-31'

#how far out do we want to predict (let's say in months for now)
prediction_windows = [3, 6, 12]

#how often is this prediction being made? every day? every month? once a year?
update_window = 12

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

for prediction_window in prediction_windows:
    # End testing at end of available time period
    test_end_time = end_time_date
    # while test_end_time is greater than or equal to start_time_date + twice the length of the prediction window (i.e. the length of a training and testing set)
    while (test_end_time >= start_time_date + 2 * relativedelta(months=+prediction_window)):
        # set start time of test x months prior to test end time, where x is the length of the prediction window
        test_start_time = test_end_time - relativedelta(months=+prediction_window)
        # set train end time equal to test start -1 day
        train_end_time = test_start_time  - relativedelta(days=+1) # minus 1 day
        # Set training start time x months prior to train end time, where x is the length of the prediction window
        train_start_time = train_end_time - relativedelta(months=+prediction_window)
        # while start time of the training set is after the start time of the whole data set
        while (train_start_time >= start_time_date):
            # print the  starting/eding times of the train and test sets
            print(train_start_time,train_end_time,test_start_time,test_end_time, prediction_window)

            train_start_time -= relativedelta(months=+prediction_window)
            # call function to get data
            #train_set, test_set = extract_train_test_sets (train_start_time, train_end_time, test_start_time, test_end_time)
            # fit on train data
            # predict on test data
        test_end_time -= relativedelta(months=+update_window)


''' OUTPUTS - train_start_time,train_end_time,test_start_time,test_end_time, prediction_window

2013-06-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2013-03-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2012-12-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2012-09-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2012-06-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2012-03-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2011-12-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2011-09-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2011-06-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2011-03-29 00:00:00 2013-09-29 00:00:00 2013-09-30 00:00:00 2013-12-31 00:00:00 3
2012-06-29 00:00:00 2012-09-29 00:00:00 2012-09-30 00:00:00 2012-12-31 00:00:00 3
2012-03-29 00:00:00 2012-09-29 00:00:00 2012-09-30 00:00:00 2012-12-31 00:00:00 3
2011-12-29 00:00:00 2012-09-29 00:00:00 2012-09-30 00:00:00 2012-12-31 00:00:00 3
2011-09-29 00:00:00 2012-09-29 00:00:00 2012-09-30 00:00:00 2012-12-31 00:00:00 3
2011-06-29 00:00:00 2012-09-29 00:00:00 2012-09-30 00:00:00 2012-12-31 00:00:00 3
2011-03-29 00:00:00 2012-09-29 00:00:00 2012-09-30 00:00:00 2012-12-31 00:00:00 3
2011-06-29 00:00:00 2011-09-29 00:00:00 2011-09-30 00:00:00 2011-12-31 00:00:00 3
2011-03-29 00:00:00 2011-09-29 00:00:00 2011-09-30 00:00:00 2011-12-31 00:00:00 3
2012-12-29 00:00:00 2013-06-29 00:00:00 2013-06-30 00:00:00 2013-12-31 00:00:00 6
2012-06-29 00:00:00 2013-06-29 00:00:00 2013-06-30 00:00:00 2013-12-31 00:00:00 6
2011-12-29 00:00:00 2013-06-29 00:00:00 2013-06-30 00:00:00 2013-12-31 00:00:00 6
2011-06-29 00:00:00 2013-06-29 00:00:00 2013-06-30 00:00:00 2013-12-31 00:00:00 6
2011-12-29 00:00:00 2012-06-29 00:00:00 2012-06-30 00:00:00 2012-12-31 00:00:00 6
2011-06-29 00:00:00 2012-06-29 00:00:00 2012-06-30 00:00:00 2012-12-31 00:00:00 6
2011-12-30 00:00:00 2012-12-30 00:00:00 2012-12-31 00:00:00 2013-12-31 00:00:00 12
'''
def temporal_train_test_sets(df, train_start, train_end, test_start, test_end, feature_cols, predictor_col):
    '''
    '''
    train_df = df[(df['date_posted'] >= train_start) & (df['date_posted'] <= train_end)]
    test_df = df[(df['date_posted'] >= test_start) & (df['date_posted'] <= test_end)]

    X_train = train_df[feature_cols]
    y_train = train_df[predictor_col]

    X_test = test_df[feature_cols]
    y_test = test_df[predictor_col]

    return X_train, y_train, X_test, y_test



