# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:58:25 2019

@author: Administrator


"""
# 获取基金类型、基金规模、分红、基金公司数据
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import VARCHAR, DECIMAL, DATE
import numpy as np
import pandas as pd
# import time
from dateutil.relativedelta import relativedelta
import multiprocessing
from multiprocessing import Pool
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
multiprocessing.freeze_support()
cpu_count = multiprocessing.cpu_count()


def argquantil(x, a):
    x = np.array(x)
    assert len(x) > 0
    argquan = []
    for i in range(len(x)):
        y = x[max((i - a), 0):(i + 1)]
        argquan.append(len(y[y <= x[i]]) / len(y))
    return argquan


def create_fund_scale_feature(tup):
    fund_scales, funds_lists, cal_dates = tup
    fund_codes = fund_scales['code'].unique()
    end_date = fund_scales['date'].max() + relativedelta(months=3)
    assert len(fund_codes) == 1
    fund_code = fund_codes[0]
    try:
        if fund_code in funds_lists['code'].unique():
            fund_info = funds_lists[funds_lists['code'] == fund_code]
            fund_scales = fund_scales.merge(cal_dates, how='outer', on='date')
            fund_scales = fund_scales.merge(fund_info[['name', 'found_date', 'date', 'class_code', 'company_name']],
                                            how='outer', on='date')
            fund_scales = fund_scales.sort_values('date', ascending=False)
            fund_scales[['name', 'found_date', 'class_code', 'company_name']] = fund_scales[
                ['name', 'found_date', 'class_code', 'company_name']].fillna(method='ffill')
            fund_scales = fund_scales.sort_values('date')
            fund_scales[['code', 'scale']] = fund_scales[['code', 'scale']].fillna(method='ffill')
            fund_scales = fund_scales[fund_scales['index'] == '1']
            fund_scales = fund_scales.drop(columns=['index'])
            fund_scales = fund_scales.dropna(subset=['code'])
            fund_scales['quantil'] = argquantil(fund_scales['scale'], 52)
            fund_scales = fund_scales[fund_scales['date'] <= end_date]
            return fund_scales
    except:
        print(fund_codes)


def generate_scale_feaures(funds_scales, funds_lists, cal_dates):
    all_scales = []
    funds_scales = funds_scales.groupby('code')
    funds_scales = [fund_scales[1] for fund_scales in funds_scales]
    pool = Pool(4)
    all_scales = pool.map(create_fund_scale_feature, map(lambda x: (x, funds_lists, cal_dates), funds_scales))
    # all_scales = map(create_fund_scale_feature, map(lambda  x: (x, funds_lists, cal_dates), funds_scales))
    new_scales = pd.concat(all_scales)
    pool.close()
    return new_scales


def scale_future_process(funds_scales):
    dates_scales = funds_scales.groupby('date').scale.sum().reset_index()
    dates_scales.columns = ['date', 'all_scale']
    dates_scales = dates_scales.sort_values('date')
    dates_scales['quantil_allscale'] = argquantil(dates_scales['all_scale'], 52 * 5)
    class_dates_scales = funds_scales.groupby(['date', 'class_code']).scale.sum().reset_index()
    class_dates_scales.columns = ['date', 'class_code', 'class_scale']
    class_dates_scales = class_dates_scales[class_dates_scales['class_scale'] != 0]
    classes_scales = []
    for fund_class in class_dates_scales['class_code'].unique():
        class_scale = class_dates_scales[class_dates_scales['class_code'] == fund_class]
        class_scale = class_scale.sort_values('date')
        class_scale['quantil_class_scale'] = argquantil(class_scale['class_scale'], 52 * 5)
        classes_scales.append(class_scale)
    classes_scales = pd.concat(classes_scales)
    classes_scales = classes_scales.merge(dates_scales, how='left', on='date')
    classes_scales['class_in_all'] = classes_scales['class_scale'] / classes_scales['all_scale']
    funds_scales = funds_scales.merge(classes_scales, how='left', on=['date', 'class_code'])
    return funds_scales


def create_fund_dividend_feature(tup):
    fund_dividends, funds_lists, cal_dates = tup
    fund_codes = fund_dividends['code'].unique()
    assert len(fund_codes) == 1
    if fund_codes[0] in funds_lists['code'].unique():

        fund_dividends = fund_dividends.merge(cal_dates, how='outer', on='date')
        fund_dividends = fund_dividends.sort_values('date')
        fund_dividends['code'] = fund_dividends['code'].ffill()
        fund_dividends = fund_dividends.dropna(subset=['code'])
        fund_dividends.loc[fund_dividends['ratio'].isnull(), 'ratio'] = 0
        for i in range(len(fund_dividends) - 1):
            if fund_dividends.iloc[i, 3] is np.nan:
                fund_dividends.iloc[i + 1, 2] += fund_dividends.iloc[i, 2]
        fund_dividends = fund_dividends[fund_dividends['index'] == '1']
        fund_dividends = fund_dividends.drop(columns=['index'])
        fund_dividends['cumsum'] = fund_dividends['ratio'].cumsum()
        fund_dividends['roll_sum'] = fund_dividends['ratio'].rolling(window=52, min_periods=1).sum()
        return fund_dividends


def generate_dividend_feaures(dividend_df, funds_lists, cal_dates):
    all_dividends = []
    dividends = dividend_df.groupby('code')
    dividends = [dividend[1] for dividend in dividends]
    pool = Pool(4)
    all_dividends = pool.map(create_fund_dividend_feature, map(lambda x: (x, funds_lists, cal_dates), dividends))
    # all_dividends = map(create_fund_dividend_feature, map(lambda x: (x, funds_lists, cal_dates), dividends))
    new_dividends = pd.concat(all_dividends)
    pool.close()
    return new_dividends


def generate_market_features(market_data, cal_dates):
    market_data = market_data.merge(cal_dates, how='outer', on='date')
    market_data = market_data.sort_values('date')
    market_data[['szzz', 'PE_TTM', 'bond10y_rate']] = market_data[['szzz', 'PE_TTM', 'bond10y_rate']].fillna(
        method='ffill')
    market_data = market_data[market_data['index'] == '1']
    market_data = market_data.drop(columns=['index'])
    market_data['bond10y_rate'] = market_data['bond10y_rate'] / 100
    market_data['bond-1/pe'] = 1 / market_data['PE_TTM'] - market_data['bond10y_rate']
    market_data['quantil_szzz'] = argquantil(market_data['szzz'], 52 * 5)
    market_data['quantil_pe'] = argquantil(market_data['PE_TTM'], 52 * 5)
    market_data['quantil_cbond'] = argquantil(market_data['bond10y_rate'], 52 * 5)
    market_data['quantil_cbond-1/pe'] = argquantil(market_data['bond-1/pe'], 52)
    return market_data


if __name__ == '__main__':
    # global pool
    # pool = multiprocessing.Pool(cpu_count - 1)
    # 连接数据库
    db = pymysql.connect(host="192.168.1.6", port=3306, user="******", passwd="******", db="dzh", charset='utf8')
    # 获取基金列表
    cursor1 = db.cursor()
    sql1 = "SELECT FSYMBOL,FDSNAME,FOUNDDATE,t1.ENDDATE,t1.CLASSCODE,KEEPERNAME,t.SECURITYID FROM TQ_FD_BASICINFO t LEFT JOIN TQ_FD_TYPE t1 \
    ON t.SECURITYID=t1.SECURITYID WHERE t1.TYPESTYLE=6 AND t.TRADPLACE != '1' AND t.FOUNDDATE != 19000101 AND t.ISSTAT=1"
    cursor1.execute(sql1)
    funds_lists = pd.DataFrame(list(cursor1.fetchall()),
                               columns=['code', 'name', 'found_date', 'class_enddate', 'class_code', 'company_name',
                                        'id'])
    funds_lists = funds_lists.astype(
        {'code': 'str', 'name': 'str', 'found_date': 'datetime64', 'class_enddate': 'datetime64', 'class_code': 'str',
         'company_name': 'str', 'id': 'str'})
    funds_lists.loc[
        funds_lists['class_enddate'] == pd.to_datetime('19000101'), 'class_enddate'] = pd.Timestamp.today().date()

    # 获取基金规模数据
    cursor2 = db.cursor()
    sql2 = "SELECT t1.FSYMBOL,t1.FDSNAME,t.REPORTDATE,t.TOTFDNAV FROM TQ_FD_ASSETPORTFOLIO t \
    LEFT JOIN TQ_FD_BASICINFO t1 ON t.SECURITYID=t1.SECURITYID WHERE t1.ISSTAT=1"
    cursor2.execute(sql2)
    funds_scales = pd.DataFrame(list(cursor2.fetchall()),
                                columns=['code', 'name', 'date', 'scale'])
    funds_scales = funds_scales.astype({'code': 'str', 'name': 'str', 'date': 'datetime64', 'scale': 'float'})

    # 获取基金分红数据
    cursor3 = db.cursor()
    sql3 = "SELECT t1.FSYMBOL,t.OUTRIGHTDATE,UNITATAXDEV,DISTNAV FROM TQ_FD_BONUS t \
    LEFT JOIN TQ_FD_BASICINFO t1 ON t.SECURITYID=t1.SECURITYID WHERE t1.TRADPLACE != '1' AND t1.FOUNDDATE != 19000101"
    cursor3.execute(sql3)
    funds_dividends = pd.DataFrame(list(cursor3.fetchall()),
                                   columns=['code', 'date', 'UNITATAXDEV', 'DISTNAV'])
    funds_dividends = funds_dividends.astype(
        {'code': 'str', 'date': 'datetime64', 'UNITATAXDEV': 'float', 'DISTNAV': 'float'})
    funds_dividends['ratio'] = funds_dividends['UNITATAXDEV'] / (
            funds_dividends['DISTNAV'] + funds_dividends['UNITATAXDEV'])
    funds_dividends = funds_dividends[['code', 'date', 'ratio']]
    funds_dividends = funds_dividends.dropna()
    # 关闭数据库连接
    db.close()
    del sql1, sql2, sql3, cursor1, cursor2, cursor3

    market_data = pd.read_excel('./data/training_data/上证综指点位、估值与十年期国债.xlsx', sheet_name='Sheet1')
    market_data.columns = ['date', 'PE_TTM', 'szzz', 'bond10y_rate']
    market_data = market_data.astype(
        {'date': 'datetime64', 'PE_TTM': 'float', 'szzz': 'float', 'bond10y_rate': 'float'})
    market_data = market_data.sort_values('date', ascending=True)

    # 设置计算时间点星期日
    ## training dates
    cal_dates = pd.date_range(pd.to_datetime('20200101').date(), pd.datetime.today().date(), freq='W-SUN')
    cal_dates = pd.DataFrame({'date': cal_dates, 'index': '1'})

    funds_scales = generate_scale_feaures(funds_scales, funds_lists, cal_dates)
    funds_scales = scale_future_process(funds_scales)
    funds_dividends = generate_dividend_feaures(funds_dividends, funds_lists, cal_dates)
    market_data = generate_market_features(market_data, cal_dates)

    basic_indicators = pd.merge(funds_scales, funds_dividends, how='left', on=['code', 'date'])
    basic_indicators = basic_indicators.merge(market_data, how='left', on='date')
    basic_indicators = basic_indicators.fillna(0)

    for column in basic_indicators.columns:
        print('%s:' % column, len(basic_indicators[column][basic_indicators[column].isnull()]))

    # basic_indicators.to_csv('./data/basic_indicators_test.csv', header=True, index=False)

    basic_indicators['date'] = basic_indicators['date'].dt.date
    basic_indicators['found_date'] = basic_indicators['found_date'].dt.date
    basic_indicators = basic_indicators.round(4)

    host = '192.168.1.6'
    port = 3306
    db = 'python'
    user = '******'
    password = '******'
    engine = create_engine(str(r"mysql+pymysql://%s:" + '%s' + "@%s/%s") % (user, password, host, db), encoding='utf8')
    try:
        basic_indicators.to_sql('fundrank_basic_indicators', con=engine, if_exists='replace', index=False,
                                dtype={'code': VARCHAR(6), 'date': DATE, 'scale': DECIMAL(20, 4), 'name': VARCHAR(50),
                                       'found_date': DATE,
                                       'class_code': VARCHAR(3), 'company_name': VARCHAR(20), 'quantil': DECIMAL(5, 4),
                                       'class_scale': DECIMAL(20, 4),
                                       'quantil_class_scale': DECIMAL(5, 4), 'all_scale': DECIMAL(20, 4),
                                       'quantil_allscale': DECIMAL(5, 4),
                                       'class_in_all': DECIMAL(5, 4), 'ratio': DECIMAL(5, 4), 'cumsum': DECIMAL(6, 4),
                                       'roll_sum': DECIMAL(6, 4),
                                       'szzz': DECIMAL(10, 4), 'PE_TTM': DECIMAL(10, 4), 'bond10y_rate': DECIMAL(5, 4),
                                       'bond-1/pe': DECIMAL(5, 4),
                                       'quantil_szzz': DECIMAL(5, 4), 'quantil_pe': DECIMAL(5, 4),
                                       'quantil_cbond': DECIMAL(5, 4),
                                       'quantil_cbond-1/pe': DECIMAL(5, 4)})
    except Exception as e:
        print(e)
