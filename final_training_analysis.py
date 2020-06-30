# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:58:21 2019

@author: Administrator
"""
import numpy as np
import pandas as pd

Next6month = pd.read_csv('../data/predict_data/Next6month_predict.csv').iloc[:, 1:]
Next1month = pd.read_csv('../data/predict_data/Next1month_predict.csv').iloc[:, 1:]
Next1quar = pd.read_csv('../data/predict_data/Next1quar_predict.csv').iloc[:, 1:]
Next1year = pd.read_csv('../data/predict_data/Next1year_predict.csv').iloc[:, 1:]


def seri_base(x):
    x = x[x.notnull()]
    return pd.Series([len(x), x.min(), x.quantile(0.25), x.median(), x.quantile(0.75), x.max(), x.mean(),
                      len(x[x > 0]) / len(x) * 100],
                     index=['count', 'min', '25%', '50%', '75%', 'max', 'mean', 'pos_rate'])


def group_analysis(arg):
    df, predict_col = arg
    all_info = seri_base(df[predict_col])
    for col in df.columns[3:]:
        select_num = int(col[col.find('@') + 1:])
        idx = np.array(df[col].argsort()[-select_num:])
        select = np.array(df[predict_col])[idx]
        if select_num == 1:
            subinfo = pd.Series(select, index=[col])
        else:
            subinfo = pd.Series([np.median(select), np.mean(select), len(select[select > 0]) / len(select) * 100],
                                index=[[col] * 3, ['median', 'mean', 'pos_rate']])
        all_info = pd.concat([all_info, subinfo], axis=0)
    return all_info


def final_train_analysis(Analysis_df, predict_col):
    Analysis_df = Analysis_df.astype({'code': 'str', 'date': 'datetime64'})
    Analysis_df['code'] = ['0' * (6 - len(code)) + code for code in Analysis_df['code']]
    Analysis_df['date'] = Analysis_df['date'].dt.date
    dates = sorted(Analysis_df['date'].unique())
    Analysis_df = Analysis_df.groupby('date')
    Analysis_df = [analysis[1] for analysis in Analysis_df]
    all_desc = pd.concat(map(group_analysis, map(lambda x: (x, predict_col), Analysis_df)), axis=1).T
    all_desc.index = dates
    return all_desc


desc_next1month = final_train_analysis(Next1month, 'Next1month')
desc_next1quar = final_train_analysis(Next1quar, 'Next1quar')
desc_next6month = final_train_analysis(Next6month, 'Next6month')
desc_next1year = final_train_analysis(Next1year, 'Next1year')

desc_next1month.to_excel('./new_training/Next1month_infomation.xlsx')
desc_next1quar.to_excel('./new_training/Next1quar_infomation.xlsx')
desc_next6month.to_excel('./new_training/Next6month_infomation.xlsx')
desc_next1year.to_excel('./new_training/Next1year_infomation.xlsx')
