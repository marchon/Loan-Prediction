# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:43:12 2017

@author: Parth Patekar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')
result=pd.DataFrame(test['Loan_ID'])

def clean(df):
    df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(),inplace=True)
    df['Credit_History'].fillna(1,inplace=True)
    df['Gender'].fillna('Male',inplace=True)
    df['Married'].fillna('Yes',inplace=True)
    df['Dependents'].fillna(0,inplace=True)
    df['Self_Employed'].fillna('No',inplace=True)
    df['LoanAmount_log'] = np.log(df['LoanAmount'])
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log(df['TotalIncome'])
    var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = df[i].factorize()[0]
        df[i] = le.fit_transform(df[i])
    return df

labels=train['Loan_Status'].values
train=clean(train)
test=clean(test)
model = RandomForestClassifier(n_estimators=25)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
model=model.fit(train[predictor_var],labels)
result['Loan_Status']=model.predict(test[predictor_var])
result.to_csv('sample_submission.csv',index=False)