import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import LabelEncoder
from random import randint

def main():
    data=pd.read_csv('C:\Users\Parth Patekar\Downloads\\train.csv')
    data=mung(data)
    var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
    data=encodeLabels(data, var_mod)
    data['Loan_ID']=1
    X=np.matrix(data[[u'Loan_ID', u'Gender', u'Married', u'Dependents', u'Education',
       u'Self_Employed', u'ApplicantIncome', u'CoapplicantIncome',
       u'LoanAmount', u'Loan_Amount_Term', u'Credit_History', u'Property_Area',
       u'LoanAmount_log', u'TotalIncome',
       u'TotalIncome_log']].values)
    m,n=X.shape
    y=np.matrix(data['Loan_Status'].values).reshape((m,1))
    crossValidate(X,y)
    
    test=pd.read_csv('C:\Users\Parth Patekar\Downloads\loan\\test.csv')
    test=mung(test)
    loan_id=test['Loan_ID'].reshape((test.shape[0],1))
    test['Loan_ID']=1
    var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
    test=encodeLabels(test, var_mod)
    test=np.matrix(test.values)
    theta_new=fmin_tnc(costFunction,theta,gradFunction,(X, y, 0))[0]
    p = predict(test, theta_new)
    result=pd.DataFrame(np.hstack((loan_id,p)), columns=['Loan_ID','Loan_Status'])
    result.to_csv('sample_submission.csv',index=False)
    
def encodeLabels(data, var_mod):
    le = LabelEncoder()
    for i in var_mod:
        data[i] = data[i].factorize()[0]
        data[i] = le.fit_transform(data[i])
    return data
    
def mung(data):
    data['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median(),inplace=True)
    data['Credit_History'].fillna(1,inplace=True)
    data['Gender'].fillna('Male',inplace=True)
    data['Married'].fillna('Yes',inplace=True)
    data['Dependents'].fillna(0,inplace=True)
    data['Self_Employed'].fillna('No',inplace=True)
    data['LoanAmount_log'] = np.log(data['LoanAmount'])
    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['TotalIncome_log'] = np.log(data['TotalIncome'])
    return data

def crossValidate(X,y):
    error=[]
    m,n=X.shape
    for i in range(1):
        theta=np.matrix(np.zeros(n))
        k=randint(0,m*8/10)
        X_test=X[k:k+m*2/10,:]
        X_train_u=X[:k,:]
        X_train_d=X[k+m*2/10:,:]
        X_train=np.vstack((X_train_u, X_train_d))
        y_test=y[k:k+m*2/10]
        y_train_u=y[:k]
        y_train_d=y[k+m*2/10:]
        y_train=np.vstack((y_train_u, y_train_d))
        theta_new=fmin_tnc(costFunction,theta,gradFunction,(X_train, y_train,1))[0]
        p = predict(X_test, theta_new)
        acc=float((p==y_test)[(p==y_test)==True].shape[1])/float(p.shape[0])
        error.append(acc)
    print "Accuracy : %s" % "{0:.3%}".format(np.mean(error))

def g(z):
    return 1 / (1 + np.exp(-z))

def costFunction(theta, X, y, lmbd=1):
    X=np.matrix(X)
    y=np.matrix(y)
    theta=np.matrix(theta)
    h=g(X*theta.T)
    cost=np.sum(np.multiply(-y,np.log(h))-np.multiply((1-y),np.log(1-h)))/m + np.sum(np.power(theta[:,1:theta.shape[1]],2))*(lmbd/2*m)
    return cost

def gradFunction(theta, X, y, lmbd=1):
    X=np.matrix(X)
    y=np.matrix(y)
    theta=np.matrix(theta)
    grad=np.zeros(n)
    h=g(X*theta.T)
    for i in range(n):
        grad[i]=np.sum(np.multiply(h-y,X[:,i]))/m
    grad=grad.reshape((15,1))
    grad[1:]=grad[1:] + theta.T[1:]*lmbd/m
    return grad

def predict(X,theta):
    X=np.matrix(X)
    theta=np.matrix(theta)
    h=g(X*theta.T)
    p=np.matrix(np.zeros((h.shape[0],1),int))
    p[h>=0.5]=1;
    p[h<0.5]=0;
    return p

if __name__ == "__main__":
    main()
