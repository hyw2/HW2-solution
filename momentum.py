import pandas as pd
from pandas_datareader import data as pdr
import os
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from os import listdir
from sklearn.utils import shuffle
from joblib import dump, load
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from time import sleep

all_tickers = ["IVV", "QQQ", "URTH", "DIA"]
ticker_names = {"IVV": "IVV", "QQQ": "QQQ", "URTH": "URTH", "DIA": "DJ"}


def deco_print(msg):
    """Decorative printing method."""
    print(f"==================>     {msg}")


def find_nearest(array, value):
    """Finds nearest index to value in array."""
    array = pd.to_datetime(array).dt.date
    value = datetime.strptime(value, '%Y-%m-%d').date()
    idx = (array - value).idxmin()
    return idx


def fetch_all_data(start_date, end_date):
    """Calls fetch_ticker_data to download new data of all five indexes using either a specified date or today."""
    all_datasets = []
    for tick in all_tickers:
        if f"{ticker_names[tick]}.csv" in listdir("data"):
            df = pd.read_csv(f"data/{ticker_names[tick]}.csv")
            if df.Date[0] > start_date or df.Date[len(df) - 1] < end_date:
                df = pdr.get_data_yahoo(tick, start=start_date, end=end_date)
            else:
                df = df.loc[find_nearest(df.Date, start_date): find_nearest(df.Date, end_date)]
            df.to_csv(f"data/{ticker_names[tick]}.csv")
        else:
            df = pdr.get_data_yahoo(tick, start=start_date, end=end_date)
            df.to_csv(f"data/{ticker_names[tick]}.csv")
        df = pd.read_csv(f"data/{ticker_names[tick]}.csv")  # some weird error when adding new dataset doesn't work
        all_datasets.append(df)
    return all_datasets


def calculate_volume_weighted_price(n, df):
    """Calculates the volume weighted average price for a stock of the last n days."""
    df["Volume"] = df["Volume"].replace(0, 1)
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['Cum_Vol_Price'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
    df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Vol']
    df['log_return'] = np.log1p(df["Close"].pct_change())
    df["Vol"] = df['log_return'].rolling(window=n).std() * np.sqrt(n)
    return df


def fit_line(n, df, name):
    """Fits a line across the VWAPs of a stock to get slope and intercept on the last n days."""
    a = []
    b = []
    for i in range(n, len(df)):
        lr = LinearRegression().fit(np.arange(n).reshape(-1, 1), df["VWAP"][i - n:i])
        a_val, b_val = lr.coef_[0], lr.intercept_
        # a_val, b_val = np.polyfit(np.arange(n), df["VWAP"][i - n:i], 1)
        a.append(a_val)
        b.append(b_val)
    # print(len(df))
    # print(len(a)+len(np.zeros(n)))
    df[name + "_a"] = np.append(np.zeros(n), np.array(a), axis=0)
    df[name + "_b"] = np.append(np.zeros(n), np.array(b), axis=0)
    return df


def build_datasets(n, N, alpha, start_date, end_date):
    """Builds the training dataset for the model."""
    deco_print("building dataset. checking for existing files")
    data_list = fetch_all_data(start_date, end_date)
    for i in range(len(data_list)):
        data_list[i] = calculate_volume_weighted_price(N, data_list[i])
        data_list[i] = fit_line(N, data_list[i], ticker_names[all_tickers[i]])
        if ticker_names[all_tickers[i]] != "IVV":
            data_list[i] = data_list[i][
                ["Date", ticker_names[all_tickers[i]] + "_a", ticker_names[all_tickers[i]] + "_b"]]
        else:
            data_list[i] = data_list[i][
                ["Date", "High", "Low", "Close", "Vol", ticker_names[all_tickers[i]] + "_a",
                 ticker_names[all_tickers[i]] + "_b"]]
        if i > 0:
            data_list[i] = data_list[i].merge(data_list[i - 1], on="Date", how="right")
        # collapse full_data list into singular dataset
    full_data = data_list[len(data_list) - 1]
    back_test_data = full_data[['Date', 'High', 'Low', 'Close']]
    full_data["Low"] = full_data["Low"][::-1].shift(1).rolling(n).min()[::-1]
    # forward rolling isn't implemented in 'rolling'
    full_data["High"] = full_data["High"][::-1].shift(1).rolling(n).max()[::-1]
    full_data["next_high"] = full_data["Close"] * (1 + alpha)
    full_data["next_low"] = full_data["Close"] * (1 - alpha)
    full_data["isHigh"] = 1 * (full_data.next_high < full_data.High)
    full_data["isLow"] = 1 * (full_data.Low < full_data.next_low)
    full_data = full_data.iloc[N:len(full_data) - 1 - n]
    back_test_data = back_test_data.iloc[N:len(back_test_data) - 1 - n]
    deco_print("number of signal 'high': " + str(np.sum(full_data.isHigh)))
    deco_print("number of signal 'low': " + str(np.sum(full_data.isLow)))
    full_data.to_csv("double_check_results.csv", index=False)
    full_data = full_data[
        ["DJ_a", 'DJ_b', 'URTH_a', 'URTH_b', 'QQQ_a', 'QQQ_b', 'Vol', 'IVV_a', 'IVV_b', 'isHigh', 'isLow']]
    train_high = full_data.drop("isLow", axis=1)
    train_low = full_data.drop("isHigh", axis=1)
    return train_high, train_low, back_test_data


def split_data(df, col, split_ratio=0.2):
    """Splits dataset off a split."""
    X_train, X_test, Y_train, Y_test = train_test_split(df.drop(col, axis=1).values, df[col].values,
                                                        test_size=split_ratio, random_state=10, stratify=df[col].values)
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
    high_fit = LogisticRegressionCV(cv=7, random_state=0, max_iter=10000, class_weight="balanced").fit(X_train_h,
                                                                                                       y_train_h)
    deco_print("Accuracy of high model is at: " + str(round(high_fit.score(X_test_h, y_test_h), 4)))
    deco_print("ROCAUC score of high model is: " + str(
        round(roc_auc_score(y_test_h, high_fit.predict_proba(X_test_h)[:, 1]), 4)))
    low_fit = LogisticRegressionCV(cv=7, random_state=0, max_iter=10000, class_weight="balanced").fit(X_train_l,
                                                                                                      y_train_l)
    deco_print("Accuracy of low model is at: " + str(round(low_fit.score(X_test_l, y_test_l), 4)))
    deco_print("ROCAUC score of low model is: " + str(
        round(roc_auc_score(y_test_l, low_fit.predict_proba(X_test_l)[:, 1]), 4)))
    return high_fit, low_fit


class Model:
    def __init__(self, alpha, N, n, lot_size, start_cash, start_date, end_date):
        deco_print(
            f"Creating model with alpha = {alpha}, N = {N}, n = {n}, start date = {start_date}, end date = {end_date}")
        self.n, self.N, self.alpha, self.lot_size, self.start_cash, self.start_date, self.end_date = n, N, alpha, lot_size, start_cash, start_date, end_date
        self.d_high, self.d_low, _ = build_datasets(self.n, self.N, self.alpha, self.start_date, self.end_date)
        self.model_high, self.model_low = train_models(self.d_high, self.d_low)

    def run_back_test(self, start, end):
        """Executes backtest and returns dict."""
        deco_print(f"Starting Backtest with lot size = {self.lot_size}, and starting cash = {self.start_cash}")
        d_high, d_low, close = build_datasets(self.n, self.N, self.alpha, start, end)
        close["pred_high"] = self.model_high.predict(d_high.drop("isHigh", axis=1))
        close["act_high"] = d_high["isHigh"]
        close["pred_low"] = self.model_low.predict(d_low.drop("isLow", axis=1))
        close["act_low"] = d_low["isLow"]
        close = close.reset_index(drop=True)
        del d_high
        del d_low
        balance = self.start_cash
        deco_print(f"Backtesting a total of {len(close) - self.n} trading days.")

        trade_ledger = {}  # columns: trade_ID, Date_created, Date_closed, Entry Price, Exit Price, Benchmark Entry, Benchmark Exit, Return on Trade, Benchmark Return
        trade_blotter = []  # trade_ID, Date_created, Action, Size, Symb, Order Price, Type, Status, Fill Price, Fill/Cancelled Date
        portfolio_ledger = []  # columns: Date, Cash, Num IVV Shares, IVV Total Value, Total Port Value
        shares = 0
        short_prices = []  # [price] a price at when the short order was given
        buy_order = []  # [price, trade ID, i], i is when the limit trade was issued
        sell_order = []  # [price, trade ID, i]
        id_counter = 0
        for i in range(len(close) - self.n):
            if i % 10 == 0:
                deco_print(f"Trading day {i}, liquid cash: ${round(balance, 2)}, total trades placed: {len(trade_blotter)}")
            current_date = close.Date[i]
            close_price = close.Close[i]
            high_price = close.High[i]
            low_price = close.Low[i]
            limit_low = close_price * (1 - self.alpha)
            limit_high = close_price * (1 + self.alpha)
            if len(buy_order) > 0:
                j = 0
                while j < len(buy_order):
                    buy_price, trade_ID, day_issued = buy_order[j][0], buy_order[j][1], buy_order[j][2]
                    if high_price > buy_price > low_price or day_issued < i - self.n:
                        if day_issued < i - self.n:
                            buy_price = close_price
                        if shares >= 0:
                            shares += self.lot_size
                            balance -= self.lot_size * buy_price
                        else:  # this buy will cover the short position
                            shares += self.lot_size
                            short_price = short_prices.pop(0)
                            balance -= self.lot_size * buy_price
                            balance += self.lot_size * (short_price - buy_price)
                        trade_blotter[trade_ID][9] = current_date
                        trade_blotter[trade_ID][8] = buy_price
                        trade_blotter[trade_ID][7] = "FILLED"
                        trade_ledger[trade_ID - 1][3] = current_date
                        trade_ledger[trade_ID - 1][5] = buy_price
                        trade_ledger[trade_ID - 1][8] = np.log(trade_ledger[trade_ID - 1][4] / buy_price)
                        trade_ledger[trade_ID - 1][7] = close_price
                        trade_ledger[trade_ID - 1][9] = np.log(close_price / trade_ledger[trade_ID - 1][6])
                        del buy_order[j]
                    else:
                        j += 1
            if len(sell_order) > 0:
                j = 0
                while j < len(sell_order):
                    sell_price, trade_ID, day_issued = sell_order[j][0], sell_order[j][1], sell_order[j][2]
                    # print(f"sell price {sell_price}, trade_id {trade_ID}, day_issued {day_issued}")
                    if high_price > sell_price > low_price or day_issued < i - self.n:
                        if day_issued < i - self.n:
                            sell_price = close_price
                        shares -= self.lot_size
                        balance += self.lot_size * sell_price
                        trade_blotter[trade_ID][9] = current_date
                        trade_blotter[trade_ID][8] = sell_price
                        trade_blotter[trade_ID][7] = "FILLED"
                        trade_ledger[trade_ID - 1][3] = current_date
                        trade_ledger[trade_ID - 1][5] = sell_price
                        trade_ledger[trade_ID - 1][8] = np.log(sell_price / trade_ledger[trade_ID - 1][4])
                        trade_ledger[trade_ID - 1][7] = close_price
                        trade_ledger[trade_ID - 1][9] = np.log(close_price / trade_ledger[trade_ID - 1][6])
                        if shares <= 0:
                            short_prices.append(sell_price)
                        del sell_order[j]
                    else:
                        j += 1
            if close["pred_high"][i]:
                # market order to buy at close price then put limit order to sell
                if shares >= 0:
                    shares += self.lot_size
                    balance -= self.lot_size * close_price
                else:  # this buy will cover the short position
                    shares += self.lot_size
                    short_price = short_prices.pop(0)
                    balance -= self.lot_size * close_price
                    balance += self.lot_size * (short_price - close_price)
                # columns: trade_ID, Date_created, Action, Size, Symb, Order Price, Type, Status, Fill Price, Fill/Cancelled Date
                trade_blotter.append(
                    [id_counter, current_date, "BUY", self.lot_size, "IVV", close_price, "MKT", "FILLED", close_price,
                     current_date])
                # columns: trade_ID, Date_created, Date_closed, Position Type, Entry Price, Exit Price, Benchmark Entry, Benchmark Exit, Return on Trade, Benchmark Return
                trade_ledger[id_counter] = [id_counter, current_date, "LONG", None, close_price, None, close_price, None, None,
                                            None]
                id_counter += 1
                trade_blotter.append(
                    [id_counter, current_date, "SELL", self.lot_size, "IVV", limit_high, "LMT", "OPEN", None, None])
                sell_order.append([limit_high, id_counter, i])  # keep id_counter same with limit orders
                id_counter += 1
            elif close["pred_low"][i]:
                # market order to sell at close price then put limit order to buy
                shares -= self.lot_size
                balance += self.lot_size * close_price
                if shares <= 0:
                    short_prices.append(close_price)
                trade_blotter.append(
                    [id_counter, current_date, "SELL", self.lot_size, "IVV", close_price, "MKT", "FILLED", close_price,
                     current_date])
                trade_ledger[id_counter] = [id_counter, current_date, "SHORT", None, close_price, None, close_price, None,
                                            None, None]
                id_counter += 1
                trade_blotter.append(
                    [id_counter, current_date, "BUY", self.lot_size, "IVV", limit_low, "LMT", "OPEN", None, None])
                buy_order.append([limit_low, id_counter, i])
                id_counter += 1
            # END OF EACH TRADING DAY
            # columns: Date, Cash, Num IVV Shares, IVV Total Value, Total Port Value
            portfolio_ledger.append(
                [current_date, balance, shares, shares * close_price, (shares * close_price) + balance])

        ledger = pd.DataFrame.from_dict(trade_ledger, orient='index',
                                        columns=["ID", "Date Created", "Position Type", "Date Closed",  "Entry Price",
                                                 "Exit Price", "Benchmark Entry", "Benchmark Exit", "Return on Trade",
                                                 "Benchmark Return"])
        blotter = pd.DataFrame(trade_blotter,
                               columns=["ID", "Date Created", "Action", "Size", "Symbol", "Order Price", "Type",
                                        "Status", "Fill Price", "Fill/Cancelled Date"])
        blotter.to_csv("trade_blotter.csv", index=False)
        ledger.to_csv("trade_ledger.csv", index=False)
        portfolio = pd.DataFrame(portfolio_ledger,
                                 columns=["Date", "Cash", "Num Shares", "Share total Value", "Total Port Value"])
        portfolio.to_csv("portfolio.csv", index=False)
        deco_print(
            f"Final Account balance at: ${round(portfolio['Total Port Value'][len(portfolio_ledger) - 1], 2)}, total trades placed: {len(ledger)}")
        return ledger, blotter, portfolio

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
