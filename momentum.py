import pandas as pd
from pandas_datareader import data as pdr
import os
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from os import listdir
from sklearn.utils import shuffle
from joblib import dump, load
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from time import sleep

all_tickers = ["IVV", "QQQ", "URTH", "DIA"]
ticker_names = {"IVV": "IVV", "QQQ": "QQQ", "URTH": "URTH", "DIA": "DJ"}


def deco_print(msg):
    """Decorative printing method."""
    print(f"==================>     {msg}")


def fetch_all_data(date=None):
    """Calls fetch_ticker_data to download new data of all five indexes using either a specified date or today."""
    if date is not None:
        curr_date = datetime.strptime(date, '%Y-%m-%d')
    else:
        curr_date = datetime.today()
    date_five_years = curr_date - relativedelta(years=5)
    curr_date = curr_date.strftime('%Y-%m-%d')
    date_five_years = date_five_years.strftime('%Y-%m-%d')

    all_datasets = []
    for tick in all_tickers:
        deco_print("ticker = " + ticker_names[tick] + ", start date = " + date_five_years + ", end date = " + curr_date)
        if ticker_names[tick] + "_" + curr_date + '.csv' in listdir("data"):
            deco_print("data already exists, loading previous data")
            df = pd.read_csv("data/" + ticker_names[tick] + "_" + curr_date + '.csv')
        else:
            # df = pdr.get_data_yahoo(tick, start=date_five_years, end=curr_date)
            with open("training_data.txt", "w") as f:
                f.write(curr_date)
            while ticker_names[tick] + "_" + curr_date + '.csv' not in listdir("data"):
                sleep(0.1)
            df = pd.read_csv("data/" + ticker_names[tick] + "_" + curr_date + ".csv")
        all_datasets.append(df)
    return all_datasets


def calculate_volume_weighted_price(n, df):
    """Calculates the volume weighted average price for a stock of the last n days."""
    df["volume"] = df["volume"].replace(0, 1)
    df['Cum_Vol'] = df['volume'].cumsum()
    df['Cum_Vol_Price'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
    df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Vol']
    df['log_return'] = np.log1p(df["close"].pct_change())
    df["Vol"] = df['log_return'].rolling(window=n).std() * np.sqrt(n)
    return df


def fit_line(n, df, name):
    """Fits a line across the VWAPs of a stock to get slope and intercept on the last n days."""
    a = []
    b = []
    for i in range(n, len(df)):
        a_val, b_val = np.polyfit(np.arange(n), df["VWAP"][i - n:i], 1)
        a.append(a_val)
        b.append(b_val)
    df[name + "_a"] = np.append(np.zeros(n), np.array(a), axis=0)
    df[name + "_b"] = np.append(np.zeros(n), np.array(b), axis=0)
    return df


def build_datasets(n, restart=False, alpha=0.01, date=None):
    """Builds the training dataset for the model."""
    if restart:
        deco_print("building dataset, restart is flagged")
        data_list = fetch_all_data(date)
        for i in range(len(data_list)):
            data_list[i] = calculate_volume_weighted_price(n, data_list[i])
            data_list[i] = fit_line(n, data_list[i], ticker_names[all_tickers[i]])
            if ticker_names[all_tickers[i]] != "IVV":
                data_list[i] = data_list[i][
                    ["date", ticker_names[all_tickers[i]] + "_a", ticker_names[all_tickers[i]] + "_b"]]
            else:
                data_list[i] = data_list[i][
                    ["date", "high", "low", "close", "Vol", ticker_names[all_tickers[i]] + "_a",
                     ticker_names[all_tickers[i]] + "_b"]]
            if i > 0:
                data_list[i] = data_list[i].merge(data_list[i - 1], on="date", how="right")
        # collapse full_data list into singular dataset
        full_data = data_list[len(data_list) - 1]
        full_data["low"] = full_data["low"].shift(-1)
        full_data["high"] = full_data["high"].shift(-1)
        full_data["next_high"] = full_data["close"] * (1 + alpha)
        full_data["next_low"] = full_data["close"] * (1 - alpha)
        full_data["isHigh"] = 1 * (full_data.next_high < full_data.high)
        full_data["isLow"] = 1 * (full_data.low < full_data.next_low)
        full_data = full_data.iloc[n:len(full_data) - 1]
        deco_print("number of signal 'a': " + str(np.sum(full_data.isHigh)))
        deco_print("number of signal 'b': " + str(np.sum(full_data.isLow)))
        full_data = full_data[
            ["DJ_a", 'DJ_b', 'URTH_a', 'URTH_b', 'QQQ_a', 'QQQ_b', 'Vol', 'IVV_a', 'IVV_b', 'isHigh', 'isLow']]
        full_data.to_csv("data/full_data_alpha_" + str(alpha) + date + ".csv", index=False)
    else:
        deco_print("loading dataset")
        full_data = pd.read_csv("data/full_data_alpha_" + str(alpha) + date + ".csv")
    train_high = full_data.drop("isLow", axis=1)
    train_low = full_data.drop("isHigh", axis=1)
    return train_high, train_low


def split_data(df, col, split_ratio=0.2):
    """Splits dataset off a split."""
    X_train, X_test, Y_train, Y_test = train_test_split(df.drop(col, axis=1).values, df[col].values, test_size=split_ratio, random_state=1, stratify=df[col].values)
    # split_index = int(len(df) * split_ratio)
    # test_set = df.iloc[:split_index].drop(col, axis=1).values
    # y_test = df.iloc[:split_index][col].values.astype(np.float)
    # train_set = df.iloc[split_index:].drop(col, axis=1).values
    # y_train = df.iloc[split_index:][col].values.astype(np.float)
    return X_train, X_test, Y_train, Y_test


def train_models(train_high, train_low):
    """Trains the two glmnet models given the two datasets."""
    deco_print("training high and low models")
    train_high, train_low = shuffle(train_high, train_low, random_state=0)
    X_train_h, X_test_h, y_train_h, y_test_h = split_data(train_high, 'isHigh')
    X_train_l, X_test_l, y_train_l, y_test_l = split_data(train_low, 'isLow')
    high_fit = LogisticRegressionCV(random_state=0, max_iter=10000, class_weight="balanced").fit(X_train_h, y_train_h)
    deco_print("Accuracy of high model is at: " + str(round(high_fit.score(X_test_h, y_test_h), 4)))
    deco_print("ROCAUC score of high model is: " + str(round(roc_auc_score(y_test_h, high_fit.predict_proba(X_test_h)[:,1]), 4)))
    low_fit = LogisticRegressionCV(random_state=0, max_iter=10000, class_weight="balanced").fit(X_train_l, y_train_l)
    deco_print("Accuracy of low model is at: " + str(round(low_fit.score(X_test_l, y_test_l), 4)))
    deco_print("ROCAUC score of low model is: " + str(round(roc_auc_score(y_test_l, low_fit.predict_proba(X_test_l)[:, 1]), 4)))
    return high_fit, low_fit


class Model:
    def __init__(self, n=5, restart=False, alpha=0.01, date=None):
        deco_print("creating model with n = "+str(n)+" restart = "+str(restart)+" and alpha = "+str(alpha) + "date" + str(date))
        self.n = n
        self.restart = restart
        self.alpha = alpha
        self.date = date
        self.d_high, self.d_low = build_datasets(self.n, self.restart, self.alpha, self.date)
        self.curr_date = datetime.today().strftime('%Y-%m-%d')
        self.model_high_name = "model_high_alpha_" + str(alpha) + "_" + self.curr_date + ".joblib"
        self.model_low_name = "model_low_alpha_" + str(alpha) + "_" + self.curr_date + ".joblib"
        if self.restart or self.model_low_name not in listdir("models") or self.model_high_name not in listdir("models"):
            self.model_high, self.model_low = train_models(self.d_high, self.d_low)
            dump(self.model_high, "models/" + self.model_high_name)
            dump(self.model_low, "models/" + self.model_low_name)
        else:
            self.model_high = load("models/" + self.model_high_name)
            self.model_low = load("models/" + self.model_low_name)


# m = Model(n=5, restart=True)

# def model_predict(model, data, p_type='class', s_type="lambda_min", actual=None):
#     """Using model, make predictions on data input and compare to actual if given."""
#     # glmnet isn't available on windows
#     # high_fit = cvglmnet(x=X_train_h.copy(), y=y_train_h.copy(), family='binomial')
#     # deco_print("high model finished training")
#     # low_fit = cvglmnet(x=X_train_l.copy(), y=y_train_l.copy(), family='binomial')
#     # high_predict = model_predict(high_fit, X_test_h, actual=y_test_h)
#     # high_predict2 = model_predict(high_fit, X_test_h, s_type="lambda_1se", actual=y_test_h)
#     # deco_print("low model finished training")
#     # low_predict = model_predict(low_fit, X_test_l, actual=y_test_l)
#     # low_predict2 = model_predict(low_fit, X_test_l, s_type="lambda_1se", actual=y_test_l)
#     predictions = cvglmnetPredict(model, newx=data, s=s_type, ptype=p_type)
#     if actual is not None:
#         deco_print("Accuracy is at: " + str(
#             round(np.sum(predictions == actual.reshape(len(actual), 1)) / len(actual), 4)) + "% with s_type= " + s_type)
#     else:
#         return predictions
