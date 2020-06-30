# -*- coding: utf-8 -*-
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import VARCHAR, DECIMAL, DATE
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from statsmodels import regression
import statsmodels.api as sm
import multiprocessing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
multiprocessing.freeze_support()
cpu_count = multiprocessing.cpu_count()

colnames = ['VolYea', 'Alpha', 'Beta', 'Annual_SharpeRatio', 'TreynorRatio', 'JensenAlpha', 'DownsideDeviation',
            'SortinoRatio', 'TrackError', 'InformationRatio']


def basic_cal(data_week, name, freerisk=0.035 / 52):
    '''
    计算如下指标：\n
    calcalate the following indexs:
    'volyea','alpha','beta','sharpe',
    'treynor','jensen','downside','sortino','trackerror','information'
    '''
    # 'volyea'
    VolYea = np.sqrt(data_week['fund_retweek'].var() * 52)
    # 'alpha','beta'
    model = regression.linear_model.OLS((data_week['fund_retweek'] - freerisk),
                                        sm.add_constant(data_week['index_retweek'] - freerisk)).fit()
    Alpha = model.params[0]
    Beta = model.params[1]
    # 'sharpe'
    if data_week['fund_retweek'].std() == 0:
        Annual_SharpeRatio = np.inf
    else:
        Annual_SharpeRatio = (data_week['fund_retweek'].mean() - freerisk) / data_week['fund_retweek'].std() * np.sqrt(
            52)
    # 'treynor'
    if Beta == 0:
        TreynorRatio = np.inf
    else:
        TreynorRatio = (data_week['fund_retweek'].mean() - freerisk) / Beta
    # 'jensen'
    JensenAlpha = data_week['fund_retweek'].mean() - freerisk + Beta * (data_week['index_retweek'].mean() - freerisk)
    # 'downside'
    DownsideDeviation = np.average([min(0, ret) ** 2 for ret in data_week['diff_ret']]) ** 0.5
    # 'sortino'  (Portfolio Return − Risk Free Rate) / Portfolio Downside Standard Deviation
    if DownsideDeviation == 0:
        SortinoRatio = np.inf
    else:
        SortinoRatio = (data_week['fund_retweek'].mean() - freerisk) / DownsideDeviation
    # 'trackerror'
    TrackError = ((data_week['diff_ret'] ** 2).sum() / (len(data_week) - 1)) ** 0.5
    # 'information'
    InformationRatio = data_week['diff_ret'].mean() / TrackError
    return pd.Series([VolYea, Alpha, Beta, Annual_SharpeRatio, TreynorRatio,
                      JensenAlpha, DownsideDeviation, SortinoRatio, TrackError, InformationRatio],
                     index=[name + colname for colname in colnames])


def bione_cal(args):
    '''
    计算单个基金在一个时间点，过往不同时间段相关指标。
    时间段可以是：past 5 years, 3 years, 2 years, 1 years, 6 months, 3 months, 2 months, 1 months, 1 weeks
    '''
    years, months, weeks, date, found_date, data, data_week = args
    name = str(sum([years, months, weeks])) + 'years' * bool(years) + 'months' * bool(months) + 'weeks' * bool(weeks)
    stardate = date - relativedelta(years=years, months=months, weeks=weeks)
    if stardate > found_date:
        data_sub = data.copy()[(data['date'] >= stardate) & (data['date'] <= date)]
        data_week_sub = data_week.copy()[(data_week['date'] >= stardate) & (data_week['date'] <= date)]

        Return = data.copy()[data['date'] <= date].iloc[-1, 1] / data.copy()[data['date'] <= stardate].iloc[-1, 1] - 1
        if len(data_sub) <= 1:
            ReturnYear = np.nan
            MaxDrop = np.nan
        else:
            ReturnYear = (data.copy()[data['date'] <= date].iloc[-1, 1] / data.copy()[data['date'] <= stardate].iloc[
                -1, 1]) ** (365 / (
                    data.copy()[data['date'] <= date].iloc[-1, 0] - data.copy()[data['date'] <= stardate].iloc[
                -1, 0]).days) - 1
            MaxDrop = ((data_sub.iloc[:, 1].cummax() - data_sub.iloc[:, 1]) / data_sub.iloc[:, 1].cummax()).max()

        info1 = pd.Series([Return, ReturnYear, MaxDrop], index=[name + 'Return', name + 'ReturnYear', name + 'MaxDrop'])

        if (date - stardate).days <= 7:
            info2 = pd.Series([np.nan] * 10, index=[name + colname for colname in colnames])
        else:
            info2 = basic_cal(data_week_sub, name)
        info = pd.concat([info1, info2])
        return info


def onetime_cal(args):
    '''
    计算单个基金在一个时间点，过往不同时间段相关指标。
    时间段包括：past 5 years, 3 years, 2 years, 1 years, 6 months, 3 months, 2 months, 1 months, 1 weeks
    '''
    data, data_week, found_date, date = args
    timeranges = [[5, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0], [0, 6, 0], [0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1]]
    info = map(bione_cal, map(lambda x: (x[0], x[1], x[2], date, found_date, data, data_week), timeranges))
    # pool = multiprocessing.Pool(cpu_count - 1)
    # info = pool.map(bione_cal, map(lambda x: (x[0], x[1], x[2], date, found_date, data, data_week), timeranges))
    # pool.close()  # 关闭进程池，不再接受新的进程
    # pool.join()  # 主进程阻塞等待子进程的退出
    info = pd.concat(info)
    info['date'] = date.date()
    '''
    计算基金未来1周、1月、1季、半年、1年的收益
    '''
    if date + relativedelta(weeks=1) <= pd.Timestamp.today():
        info['Next1week'] = data[data['date'] <= (date + relativedelta(weeks=1))]['value'].iloc[-1] / \
                            data[data['date'] <= date]['value'].iloc[-1] - 1
    else:
        info['Next1week'] = np.nan
    if date + relativedelta(months=1) <= pd.Timestamp.today():
        info['Next1month'] = data[data['date'] <= (date + relativedelta(months=1))]['value'].iloc[-1] / \
                             data[data['date'] <= date]['value'].iloc[-1] - 1
    else:
        info['Next1month'] = np.nan
    if date + relativedelta(months=3) <= pd.Timestamp.today():
        info['Next1quar'] = data[data['date'] <= (date + relativedelta(months=3))]['value'].iloc[-1] / \
                            data[data['date'] <= date]['value'].iloc[-1] - 1
    else:
        info['Next1quar'] = np.nan
    if date + relativedelta(months=6) <= pd.Timestamp.today():
        info['Next6month'] = data[data['date'] <= (date + relativedelta(months=6))]['value'].iloc[-1] / \
                             data[data['date'] <= date]['value'].iloc[-1] - 1
    else:
        info['Next6month'] = np.nan
    if date + relativedelta(years=1) <= pd.Timestamp.today():
        info['Next1year'] = data[data['date'] <= (date + relativedelta(years=1))]['value'].iloc[-1] / \
                            data[data['date'] <= date]['value'].iloc[-1] - 1
    else:
        info['Next1year'] = np.nan

    return info


def combin(one_fund, szzz_data):
    one_fund = one_fund[['date', 'value']]
    one_fund = pd.merge(one_fund, szzz_data, how='outer', on='date')
    one_fund = one_fund.sort_values('date')
    one_fund = one_fund.fillna(method='ffill')
    one_fund = one_fund.dropna()

    fund_week = one_fund.copy()
    fund_week['weekday'] = fund_week['date'].dt.weekday
    fund_week['week'] = fund_week['date'].dt.week
    fund_week['year'] = fund_week['date'].dt.year
    subline = fund_week.groupby(['year', 'week']).aggregate({'weekday': 'max'}).reset_index()
    fund_week = pd.merge(fund_week, subline, how='right', on=['year', 'week', 'weekday'])[['date', 'value', 'szzz']]
    fund_week['fund_retweek'] = fund_week['value'].pct_change()
    fund_week['index_retweek'] = fund_week['szzz'].pct_change()
    fund_week['diff_ret'] = fund_week['fund_retweek'] - fund_week['index_retweek']
    fund_week = fund_week.iloc[1:, :]
    return one_fund, fund_week


def fund_cal(args):
    '''
    针对一个基金计算各个测试时间点的数据。
    '''
    one_fund, szzz_data, cal_date = args
    fund_code = one_fund['code'].unique()
    found_date = one_fund['date'].min()
    if found_date <= pd.to_datetime('20190101'):
        one_fund, fund_week = combin(one_fund, szzz_data)
        cal_date_sub = cal_date[cal_date >= (found_date + relativedelta(years=+1))]
        info = map(onetime_cal, map(lambda x: (one_fund, fund_week, found_date, x), cal_date_sub))
        # pool = multiprocessing.Pool(cpu_count - 1)
        # info = pool.map(onetime_cal, map(lambda x: (one_fund, fund_week, found_date, x), cal_date_sub))
        # pool.close()  # 关闭进程池，不再接受新的进程
        # pool.join()  # 主进程阻塞等待子进程的退出
        info = pd.concat(info, axis=1).T
        info['code'] = fund_code[0]
        info['found_date'] = found_date.date()
        return info
    else:
        print('%s成立于%s' % (fund_code, fund_data))


if __name__ == '__main__':
    # global pool
    # pool = multiprocessing.Pool(cpu_count - 1)
    # 连接数据库
    db = pymysql.connect(host="192.168.1.6", port=3306, user="******", passwd="******", db="dzh", charset='utf8')
    # 基金净值
    cursor1 = db.cursor()
    sql1 = "SELECT t1.fsymbol,t.ENDDATE,t.REPAIRUNITNAV FROM TQ_FD_DERIVEDN t LEFT JOIN TQ_FD_BASICINFO t1 ON t.SECURITYID=t1.SECURITYID \
            INNER jOIN TQ_FD_TYPE t2 ON t.SECURITYID=t2.SECURITYID WHERE t1.TRADPLACE != '1' and t1.FOUNDDATE != 19000101 \
            AND t2.TYPESTYLE=6 and t2.CLASSCODE IN (611,621,622,634) AND t2.ENDDATE = 19000101 ORDER BY t.ENDDATE"
    cursor1.execute(sql1)
    fund_data = pd.DataFrame(list(cursor1.fetchall()), columns=['code', 'date', 'value'])
    fund_data = fund_data.astype({'code': 'str', 'date': 'datetime64', 'value': 'float'})

    # 获取上证指数
    cursor2 = db.cursor()
    sql2 = "SELECT t.TRADEDATE,t.TCLOSE FROM TQ_QT_INDEX t LEFT JOIN TQ_IX_BASICINFO t1 ON t.SECODE=t1.SECODE WHERE t1.SYMBOL='000001' ORDER BY t.TRADEDATE"
    cursor2.execute(sql2)
    szzz_data = pd.DataFrame(list(cursor2.fetchall()), columns=['date', 'szzz'])
    szzz_data = szzz_data.astype({'date': 'datetime64', 'szzz': 'float'})
    # 关闭数据库连接
    db.close()
    del sql1, sql2, cursor1, cursor2, db
    # 设置计算时间点星期日
    cal_date = pd.date_range(pd.to_datetime('20020101').date(), pd.datetime.today().date(), freq='W-SUN')

    all_info = []
    df_funds = fund_data.groupby('code')
    df_funds = [df_fund[1] for df_fund in df_funds]
    del fund_data
    all_info = map(fund_cal, map(lambda x: (x, szzz_data, cal_date), df_funds))
    # all_info = pool.map(fund_cal, map(lambda x: (x, szzz_data, cal_date), df_funds))
    all_info = pd.concat(all_info)
    # pool.join()  # 主进程阻塞等待子进程的退出
    # pool.close()  # 关闭进程池，不再接受新的进程
    # all_info.to_csv('fund_all_info_test.csv', index=False, header=True)

    all_info['date'] = all_info['date'].dt.date
    all_info['found_date'] = all_info['found_date'].dt.date
    all_info = all_info.round(4)

    columns = all_info.columns.to_list()
    columns1 = ['code', 'date', 'found_date', 'Next1month', 'Next1quar', 'Next1week', 'Next1year', 'Next6month']
    columns2 = list(set(columns) - set(columns1))
    columns2_value = [(col[:7], col[7:]) if col[1:7] == 'months' else (col[:6], col[6:]) for col in columns2]
    columns_dict = dict(zip(columns2, columns2_value))
    all_info = all_info.set_index(columns1)
    all_info.columns = pd.MultiIndex.from_tuples(columns2_value)
    all_info = all_info.stack(0).reset_index().rename(columns={'level_8': 'MeasureTime'})
    all_info = all_info.replace([np.inf, -np.inf], np.nan)

    host = '192.168.1.6'
    port = 3306
    db = 'python'
    user = '******'
    password = '******'
    engine = create_engine(str(r"mysql+pymysql://%s:" + '%s' + "@%s/%s") % (user, password, host, db), encoding='utf8')
    try:
        all_info.to_sql('fundrank_quantify_indicators', con=engine, if_exists='replace', index=False,
                        dtype={'code': VARCHAR(6), 'date': DATE, 'found_date': DATE, 'Next1month': DECIMAL(6, 4),
                               'Next1quar': DECIMAL(6, 4),
                               'Next1week': DECIMAL(6, 4), 'Next1year': DECIMAL(6, 4), 'Next6month': DECIMAL(6, 4),
                               'MeasureTime': VARCHAR(10),
                               'Alpha': DECIMAL(50, 4), 'Annual_SharpeRatio': DECIMAL(50, 4), 'Beta': DECIMAL(50, 4),
                               'DownsideDeviation': DECIMAL(50, 4),
                               'InformationRatio': DECIMAL(50, 4), 'JensenAlpha': DECIMAL(50, 4),
                               'MaxDrop': DECIMAL(50, 4), 'Return': DECIMAL(50, 4),
                               'ReturnYear': DECIMAL(50, 4), 'SortinoRatio': DECIMAL(50, 4),
                               'TrackError': DECIMAL(50, 4), 'TreynorRatio': DECIMAL(50, 4),
                               'VolYea': DECIMAL(50, 4)})
    except Exception as e:
        print(e)
