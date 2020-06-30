# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:03:20 2019

@author: Administrator
"""
import pymysql
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import warnings
warnings.simplefilter(action='ignore', category='FutureWarning')


def date_change(x):
    if x == '19000101':
        x = '21000101'
    return x


db = pymysql.connect(host="118.31.72.134", port=3306, user="taoji", passwd="Abc12345", db="dzh", charset='utf8')
cursor = db.cursor()
sql = "SELECT FSYMBOL,ENDDATE FROM TQ_FD_BASICINFO"
cursor.execute(sql)
fund_endate = pd.DataFrame(list(cursor.fetchall()), columns=['code', 'enddate'])
fund_endate['enddate'] = [date_change(date) for date in fund_endate['enddate']]
fund_endate = fund_endate.astype({'code': 'str', 'enddate': 'datetime64'})
db.close()
del sql, cursor

quaninfo = pd.read_csv('../data/fund_quaninfo.csv')
basiinfo = pd.read_csv('../data/basic_indicators_df.csv')


def data_preprocess(basiinfo, quaninfo, fund_endate):
    quaninfo['code'] = quaninfo['code'].astype('str')
    quaninfo['code'] = [('0' * (6 - len(code)) + code) for code in quaninfo['code']]
    quaninfo[['date', 'found_date']] = quaninfo[['date', 'found_date']].astype('datetime64')
    quaninfo = quaninfo[quaninfo['date'] >= pd.to_datetime('20050101')]

    basiinfo = basiinfo.astype(
        {'code': 'str', 'date': 'datetime64', 'name': 'str', 'found_date': 'datetime64', 'class_code': 'str',
         'company_name': 'str'})
    basiinfo['code'] = [('0' * (6 - len(code)) + code) for code in basiinfo['code']]

    basiinfo['class_code'] = pd.Categorical(basiinfo['class_code'])
    basiinfo['company_name'] = pd.Categorical(basiinfo['company_name'])
    basiinfo = basiinfo.merge(fund_endate, how='left', on='code')
    basiinfo = basiinfo[basiinfo['date'] <= basiinfo['enddate']].drop(columns='enddate')
    info_df = basiinfo.merge(quaninfo, how='left', on=['code', 'date', 'found_date'])

    info_df['found_year'] = pd.Categorical(info_df['found_date'].dt.year)
    info_df['found_month'] = pd.Categorical(info_df['found_date'].dt.month)
    info_df['found_day'] = pd.Categorical(info_df['found_date'].dt.day)
    info_df['found_date_delt'] = (info_df['date'] - info_df['found_date'])
    info_df['found_date_delt'] = [days.days for days in info_df['found_date_delt']]
    ############ 成立时间筛选
    # info_df = info_df[info_df['found_date_delt'] >= 365]
    ############ 删除其它 nan ，inf特征 ##########################################
    dropcolnames = ['name', 'found_date']
    info_df = info_df.drop(columns=dropcolnames)
    ###########  基金类型是否筛选
    info_df = info_df[info_df['class_code'].isin(['611', '621'])]
    info_df = info_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, thresh=100)

    lb_class = LabelBinarizer()
    info_df['class_code'] = lb_class.fit_transform(info_df['class_code'])[:, 0]

    lb_company = LabelBinarizer()
    companys_code = pd.DataFrame(lb_company.fit_transform(info_df['company_name']), columns=list(
        map(lambda x: 'company_code' + str(x), range(len(info_df['company_name'].unique())))))
    lb_year = LabelBinarizer()
    years_code = pd.DataFrame(lb_year.fit_transform(info_df['found_year']), columns=list(
        map(lambda x: 'year_code' + str(x), range(len(info_df['found_year'].unique())))))
    lb_month = LabelBinarizer()
    months_code = pd.DataFrame(lb_month.fit_transform(info_df['found_month']), columns=list(
        map(lambda x: 'month_code' + str(x), range(len(info_df['found_month'].unique())))))
    lb_day = LabelBinarizer()
    days_code = pd.DataFrame(lb_day.fit_transform(info_df['found_day']), columns=list(
        map(lambda x: 'day_code' + str(x), range(len(info_df['found_day'].unique())))))

    info_df = pd.concat([info_df.reset_index(drop=True), companys_code, years_code, months_code, days_code], axis=1)
    info_df = info_df.drop(columns=['company_name', 'found_year', 'found_month', 'found_day'])
    return info_df


info_df = data_preprocess(basiinfo, quaninfo, fund_endate)
del quaninfo, basiinfo, fund_endate


def data_split(data, predict_column, train_ratio=0.6):
    data = data.sort_values('date')
    assert train_ratio <= 1
    n_sample = len(data)
    n_train_min = int(n_sample * train_ratio)
    n_train = len(data[data['date'] <= (data['date'].iloc[n_train_min])])
    n_valid = len(data[data['date'] <= (data['date'].iloc[(int((n_sample + n_train) / 2))])])
    del n_train_min

    colnames = ['MaxDrop', 'Return', 'ReturnYear', 'VolYea', 'Alpha', 'Beta', 'Annual_SharpeRatio', 'TreynorRatio',
                'JensenAlpha', 'DownsideDeviation', 'SortinoRatio', 'TrackError', 'InformationRatio']
    timelist = ['1months', '2months', '3months', '6months', '1years', '2years', '3years']
    norm_col = sum(list(map(lambda y: list(map(lambda x: y + x, colnames)), timelist)), [])
    norm_col.extend(
        ['scale', 'quantil', 'class_scale', 'quantil_class_scale', 'all_scale', 'quantil_allscale', 'class_in_all',
         'ratio', 'cumsum', 'roll_sum',
         'szzz', 'PE_TTM', 'bond10y_rate', 'bond-1/pe', 'quantil_szzz', 'quantil_pe', 'quantil_cbond',
         'quantil_cbond-1/pe', 'found_date_delt', ])
    data_sub = data[norm_col]

    norm = StandardScaler()
    norm_data_sub1 = norm.fit_transform(data_sub.iloc[:n_train, :])
    norm_data_sub2 = norm.transform(data_sub.iloc[n_train:, :])
    norm_data_sub = np.concatenate([norm_data_sub1, norm_data_sub2], axis=0)
    norm_df = pd.DataFrame(norm_data_sub, columns=norm_col)

    drop_df = data[list(set(data.columns) - set(norm_col))]
    norm_df = pd.concat([drop_df.reset_index(drop=True), norm_df], axis=1)
    del norm_data_sub1, norm_data_sub2, norm_data_sub, data, data_sub
    norm_df[predict_column] = norm_df[predict_column] * 100

    train_df = norm_df.iloc[:n_train, :]
    x_train = train_df.drop(columns=[predict_column, 'code', 'date']).values
    # y_train = train_df[predict_column].groupby(train_df['date']).rank(ascending=False).astype('int').values.flatten()
    y_train = train_df[predict_column].astype('int').values.flatten()
    group_train = train_df.groupby('date').date.count().sort_index(ascending=True).values.flatten()

    valid_df = norm_df.iloc[n_train:n_valid, :]
    x_valid = valid_df.drop(columns=[predict_column, 'code', 'date']).values
    # y_valid = valid_df[predict_column].groupby(valid_df['date']).rank(ascending=False).astype('int').values.flatten()
    y_valid = valid_df[predict_column].astype('int').values.flatten()
    group_valid = valid_df.groupby('date').date.count().sort_index(ascending=True).values.flatten()

    test_df = norm_df.iloc[n_valid:, :]
    x_test = test_df.drop(columns=[predict_column, 'code', 'date']).values
    y_true = test_df[['code', 'date', predict_column]]

    return x_train, y_train, group_train, x_valid, y_valid, group_valid, x_test, y_true, norm


def training_predicting(x_train, y_train, group_train, x_valid, y_valid, group_valid, x_test, eval_metric, select_num):
    param = {'verbosity': 2, 'booster': 'gblinear', 'n_jobs': -1}
    param['objective'] = 'rank:%s' % eval_metric
    search_param = {'n_estimators': [100, 200, 500, 1000],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.1, 0.05, 0.025, 0.01, 0.001],
                    'gamma': [0, 0.001, 0.01, 0.1, 0.2, 0.5, 1],
                    'reg_alpha': [0, 0.001, 0.01, 0.1, 0.2, 0.5, 1],
                    'reg_lambda': [0, 0.001, 0.01, 0.1, 0.2, 0.5, 1]}

    best_eval_score = np.inf
    for i in range(100):
        params = {k: np.random.choice(v) for k, v in search_param.items()}
        params = dict(param, **params)
        model = xgb.XGBRanker(**params)
        model.fit(X=x_train, y=y_train, group=group_train,
                  eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_group=[group_train, group_valid],
                  eval_metric='%s@%s' % (eval_metric, select_num), verbose=True)
        score = model.evals_result['eval_1']['%s@%s' % (eval_metric, select_num)][-1]
        if score < best_eval_score:
            best_eval_score = score
            # best_params1 = params
            best_model = model
    print('ndcg_best_score', best_eval_score)
    predict = best_model.predict(x_test)
    return best_model, predict


def predict_col_tuning(predict_column, info_df=info_df):
    predict_columns = ['Next1week', 'Next1month', 'Next1quar', 'Next6month', 'Next1year']
    predict_columns.remove(predict_column)
    info_df = info_df.drop(columns=predict_columns)
    info_df = info_df[info_df[predict_column].notnull()]
    x_train, y_train, group_train, x_valid, y_valid, group_valid, x_test, y_true, norm = data_split(info_df,
                                                                                                    predict_column)
    model_dict = dict()
    for eval_metric in ['ndcg', 'map']:
        for select_num in [1, 2, 3, 4, 5, 7, 10]:
            model, predict = training_predicting(x_train, y_train, group_train, x_valid, y_valid, group_valid, x_test,
                                                 eval_metric, select_num)
            y_true['%s@%s' % (eval_metric, select_num)] = predict
            model_dict.update({(eval_metric, select_num): model})
            model.save_model('./new_training/%s_%s@%s' % (predict_column, eval_metric, select_num))
    y_true.to_csv('./new_training/%s_predict.csv' % predict_column)
    return y_true, model_dict


if __name__ == '__main__':
    true_dicts = dict()
    model_dicts = dict()
    for col in ['Next1month', 'Next1quar', 'Next6month', 'Next1year'][1]:
        y_true, model_dict = predict_col_tuning(col, info_df)
        true_dicts.update({col: y_true})
        model_dicts.update({col: model_dict})
