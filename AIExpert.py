# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:06:53 2023

@author: 3ndalib
"""

import pandas as pd
from pandas.api.types import is_object_dtype as ObjectDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
from sklearn.linear_model import LinearRegression as LNR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import SGDRegressor as sgdr
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier as KNC, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier as DTC,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier as RFC ,RandomForestRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import StandardScaler as SS
from sklearn.metrics import mean_absolute_error,accuracy_score,precision_score,recall_score,r2_score,mean_squared_error


pd.set_option('display.max_columns', None)
    
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

file = st.file_uploader(
    "Upload your data file",
    accept_multiple_files=False,
    key=st.session_state["file_uploader_key"],
)

if file:
    st.session_state["uploaded_files"] = file

# if st.button("Clear uploaded files"):
#     st.session_state["file_uploader_key"] += 1
#     st.experimental_rerun()



if file is not None:
        extension = file.name.split(".")[-1]
        if(extension=="csv"):
            df = pd.read_csv(file)
            st.toast("File uploaded succesfuly")
        elif(extension=="xls" or extension=="xlsx"):
            df = pd.read_excel(file)
            st.toast("File uploaded succesfuly")
        elif(extension=="sql"):
            df = pd.read_sql(file)
            st.toast("File uploaded succesfuly")
        else:
            st.session_state["file_uploader_key"] += 1
            st.experimental_rerun()  
            

def TaskType(df,target):
    if(not (df[target].dtype.kind in 'iufc')):
        return "classification"
    else:
        if(df[target].nunique()>5):
            return "regression"
        else:
            return "classification"

def DataInfo(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return (df.head(10), df.describe(), s, df.nunique())

def HandleNan(df,numerical,categorical):
    for i in range(df.shape[1]):
        if( df.iloc[:,i].dtype.kind in 'iufc'):
            if(numerical=="mean"):
                df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].mean())
            elif(numerical=="median"):
                df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].median())
            else:
                df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].mode()[0])
        else:
            if(categorical=="mode"):
                df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].mode()[0])
            else:
                df.iloc[:,i] = df.iloc[:,i].fillna(categorical)
    return df
def LabelEncoding(df):
    for i in range(df.shape[1]):
        if(not (df.iloc[:,i].dtype.kind in 'iufc')):
            df.iloc[:,i] = LE().fit_transform(df.iloc[:,i])
            df.iloc[:,i] = df.iloc[:,i].astype('int')
    return df

def StandardScaling(df,target=None):
    for i in range(df.shape[1]):
        if(df.iloc[:,i].dtype.kind in 'biufc' and df.columns[i]!=target):
            df.iloc[:,i] = SS().fit_transform(df[[df.columns[i]]]).flatten()
    return df
def PreProcess(df,target,numerical,categorical):
    st.write("HandleNan")
    df = HandleNan(df,numerical,categorical)
    st.write(df)
    st.write("Encoding")
    df = LabelEncoding(df)
    st.write(df)
    if(target!=None):
        df[target]=df[target].astype('int')
    return(df)    

    
class ClassificationExpert:
    def setup(self,data,target,numerical,categorical,TestSize=0.3,xpred = None):
        self.data = data
        self.target = target
        self.numerical = numerical
        self.categorical = categorical
        self.TestSize = TestSize
        self.PPData = PreProcess(self.data,self.target,self.numerical,self.categorical)

        self.x,self.y=self.PPData.drop(target,axis=1),self.PPData[target]

        self.XTrain,self.XTest,self.YTrain,self.YTest =  tts(self.x,self.y,test_size=TestSize)
        self.info = pd.DataFrame({"Description":["Target",
                                                  "Original Shape",
                                                  "Shape after preprocessing",
                                                  "XTrainShape",
                                                  "YTrainShape",
                                                  "XTestShape",
                                                  "YTestShape"],
                                  "Value":[target
                                            , data.shape,
                                            self.PPData.shape,
                                            self.XTrain.shape,
                                            self.YTrain.shape,
                                            self.XTest.shape,
                                            self.YTest.shape]})
    def predict(self,xpred):
        self.lrxpreds = self.lrclf.predict(xpred)
        self.csvxpreds = self.csvclf.predict(xpred)
        self.knxpreds = self.knclf.predict(xpred)
        self.dtxpreds = self.dtclf.predict(xpred)
        self.rfxpreds = self.rfclf.predict(xpred)
        
        self.ModelsResults = pd.DataFrame(
            {"Model":[
                "LogisticRegression",
                "SuppportVectorMachine",
                "KNeighrestNeighbor",
                "DecisionTree",
                "RandomForest",],
             "Prediction":[ 
                 self.lrxpreds,
                 self.csvxpreds,
                 self.knxpreds,
                 self.dtxpreds,
                 self.rfxpreds,],})
        return self.ModelsResults
    def CompareModels(self):
        self.lrclf = LR()

        self.lrclf.fit(self.XTrain, self.YTrain)
        self.lrpreds = self.lrclf.predict(self.XTrain)
        self.lrtestpreds = self.lrclf.predict(self.XTest)
       
        self.csvclf = SVC()
        self.csvclf.fit(self.XTrain, self.YTrain)
        self.csvpreds = self.csvclf.predict(self.XTrain)
        self.csvtestpreds = self.csvclf.predict(self.XTest)
        
        self.knclf = KNC()
        self.knclf.fit(self.XTrain, self.YTrain)
        self.knpreds = self.knclf.predict(self.XTrain.values)
        self.kntestpreds = self.knclf.predict(self.XTest.values)
        
        self.dtclf = DTC(max_depth=5)
        self.dtclf.fit(self.XTrain, self.YTrain)
        self.dtpreds = self.dtclf.predict(self.XTrain)
        self.dttestpreds = self.dtclf.predict(self.XTest)
        
        self.rfclf = RFC(n_estimators=20)
        self.rfclf.fit(self.XTrain, self.YTrain)
        self.rfpreds = self.rfclf.predict(self.XTrain)
        self.rftestpreds = self.rfclf.predict(self.XTest)
        
        self.ModelsPerformance = pd.DataFrame(
            {"Model":[
                "LogisticRegression",
                "SuppportVectorMachine",
                "KNeighrestNeighbor",
                "DecisionTree",
                "RandomForest",],
             "TrainAccuracy":[ 
                 accuracy_score(self.YTrain, self.lrpreds),
                 accuracy_score(self.YTrain, self.csvpreds),
                 accuracy_score(self.YTrain, self.knpreds),
                 accuracy_score(self.YTrain, self.dtpreds),
                 accuracy_score(self.YTrain, self.rfpreds),],
             "TestAccuracy":[ 
                 accuracy_score(self.YTest, self.lrtestpreds),
                 accuracy_score(self.YTest, self.csvtestpreds),
                 accuracy_score(self.YTest, self.kntestpreds),
                 accuracy_score(self.YTest, self.dttestpreds),
                 accuracy_score(self.YTest, self.rftestpreds),],
             "Precision":[
                 precision_score(self.YTest, self.lrtestpreds, average='weighted'),
                 precision_score(self.YTest, self.csvtestpreds, average='weighted'),
                 precision_score(self.YTest, self.kntestpreds, average='weighted'),
                 precision_score(self.YTest, self.dttestpreds, average='weighted'),
                 precision_score(self.YTest, self.rftestpreds, average='weighted'),],
             "Recall":[                                            
                 recall_score(self.YTest, self.lrtestpreds, average='macro'),
                 recall_score(self.YTest, self.csvtestpreds, average='macro'),
                 recall_score(self.YTest, self.kntestpreds, average='macro'),
                 recall_score(self.YTest, self.dttestpreds, average='macro'),
                 recall_score(self.YTest, self.rftestpreds, average='macro'),]})
        return(self.ModelsPerformance)
    #     self.best = self.ModelsPerformance.query('TestAccuracy == TestAccuracy.max()')
    #     self.best = self.ModelsPerformance.query('TrainAccuracy == TrainAccuracy.max()')

    # def plot(self):
    #     disp = ConfusionMatrixDisplay.from_estimator(self.rfclf,self.XTrain,self.YTrain)
    #     disp = ConfusionMatrixDisplay.from_estimator(self.rfclf,self.XTest,self.YTest)

class RegressionExpert:
    def setup(self,data,target,numerical,categorical,TestSize=0.3):
        self.data = data
        self.target = target
        self.numerical = numerical
        self.categorical = categorical
        self.TestSize = TestSize
        self.PPData = PreProcess(self.data,self.target,self.numerical,self.categorical)

        self.x,self.y=self.PPData.drop(target,axis=1),self.PPData[target]
        
        self.XTrain,self.XTest,self.YTrain,self.YTest =  tts(self.x,self.y,test_size=TestSize)
        self.info = pd.DataFrame({"Description":["Target",
                                                  "Original Shape",
                                                  "Shape after preprocessing",
                                                  "XTrainShape",
                                                  "YTrainShape",
                                                  "XTestShape",
                                                  "YTestShape"],
                                  "Value":[target
                                            , data.shape,
                                            self.PPData.shape,
                                            self.XTrain.shape,
                                            self.YTrain.shape,
                                            self.XTest.shape,
                                            self.YTest.shape]})
    def predict(self,xpred):
        self.lnrxpreds = self.lnr.predict(xpred)
        self.csvxpreds = self.csvr.predict(xpred)
        self.knxpreds = self.knr.predict(xpred)
        self.dtxpreds = self.dtr.predict(xpred)
        self.rfxpreds = self.rfr.predict(xpred)
        self.sgdpolyxpreds = self.sgdpoly.predict(xpred)
        
        self.ModelsResults = pd.DataFrame(
            {"Model":[
                "LinearRegression",
                "SuppportVectorMachine",
                "KNeighrestNeighbor",
                "DecisionTree",
                "RandomForest",
                "GradientDescent",
                ],
             "Prediction":[ 
                self.lnrxpreds,
                self.csvxpreds,
                self.knxpreds,
                self.dtxpreds,
                self.rfxpreds,
                self.sgdpolyxpreds,],})
        return self.ModelsResults
    def CompareModels(self):
        self.lnr = LNR()
        self.lnr.fit(self.XTrain, self.YTrain)
        self.lnrpreds = self.lnr.predict(self.XTrain)
        self.lnrtestpreds = self.lnr.predict(self.XTest)
       
        self.csvr = SVR()
        self.csvr.fit(self.XTrain, self.YTrain)
        self.csvpreds = self.csvr.predict(self.XTrain)
        self.csvtestpreds = self.csvr.predict(self.XTest)
        
        self.knr = KNeighborsRegressor()
        self.knr.fit(self.XTrain, self.YTrain)
        self.knpreds = self.knr.predict(self.XTrain.values)
        self.kntestpreds = self.knr.predict(self.XTest.values)
        
        self.dtr = DecisionTreeRegressor(max_depth=5)
        self.dtr.fit(self.XTrain, self.YTrain)
        self.dtpreds = self.dtr.predict(self.XTrain)
        self.dttestpreds = self.dtr.predict(self.XTest)
        
        self.rfr = RandomForestRegressor(n_estimators=20)
        self.rfr.fit(self.XTrain, self.YTrain)
        self.rfpreds = self.rfr.predict(self.XTrain)
        self.rftestpreds = self.rfr.predict(self.XTest)
        
        self.sgdpoly = sgdr(max_iter=100)
        self.sgdpoly.fit(self.XTrain, self.YTrain)
        self.polysgdpreds = self.sgdpoly.predict(self.XTrain)
        self.polysgdtestpreds = self.sgdpoly.predict(self.XTest)
        
        self.ModelsPerformance = pd.DataFrame(
            {"Model":[
                "LinearRegression",
                "SuppportVectorMachine",
                "KNeighrestNeighbor",
                "DecisionTree",
                "RandomForest",
                "GradientDescent",
                ],
             "Trainr2":[ 
                 r2_score(self.YTrain, self.lnrpreds),
                 r2_score(self.YTrain, self.csvpreds),
                 r2_score(self.YTrain, self.knpreds),
                 r2_score(self.YTrain, self.dtpreds),
                 r2_score(self.YTrain, self.rfpreds),
                 r2_score(self.YTrain, self.polysgdpreds),
                 ],
             "Testr2":[ 
                 r2_score(self.YTest, self.lnrtestpreds),
                 r2_score(self.YTest, self.csvtestpreds),
                 r2_score(self.YTest, self.kntestpreds),
                 r2_score(self.YTest, self.dttestpreds),
                 r2_score(self.YTest, self.rftestpreds),
                 r2_score(self.YTest, self.polysgdtestpreds),
                 ],
              "MeanSquareError":[
                 mean_squared_error(self.YTest, self.lnrtestpreds),
                 mean_squared_error(self.YTest, self.csvtestpreds),
                 mean_squared_error(self.YTest, self.kntestpreds),
                 mean_squared_error(self.YTest, self.dttestpreds),
                 mean_squared_error(self.YTest, self.rftestpreds),
                 mean_squared_error(self.YTest, self.polysgdtestpreds),
             ],
              "MeanAbsoluteError":[                                            
                 mean_absolute_error(self.YTest, self.lnrtestpreds),
                 mean_absolute_error(self.YTest, self.csvtestpreds),
                 mean_absolute_error(self.YTest, self.kntestpreds),
                 mean_absolute_error(self.YTest, self.dttestpreds),
                 mean_absolute_error(self.YTest, self.rftestpreds),
                 mean_absolute_error(self.YTest, self.polysgdtestpreds),
             ]
             })
        return(self.ModelsPerformance)
        # self.best = self.ModelsPerformance.query('TestAccuracy == TestAccuracy.max()')
        # self.best = self.ModelsPerformance.query('TrainAccuracy == TrainAccuracy.max()')
        # print(self.best)
    # def plot(self):
    #     disp = ConfusionMatrixDisplay.from_estimator(self.rfclf,self.XTrain,self.YTrain)
    #     disp = ConfusionMatrixDisplay.from_estimator(self.rfclf,self.XTest,self.YTest)

if file is not None:
    h,d,i,n = DataInfo(df)
    st.caption("Data head")
    st.write(h)
    st.caption("Data describtion")
    st.write(d)
    st.caption("Data info")
    st.text(i)
    st.caption("Data unique values")
    st.write(n)
    

    target = st.selectbox('Choose target column to predict', df.columns,key="adfsdaf")
    d = df.columns.values.tolist()
    d.remove(target)
    dropcolumns = st.multiselect('Select Columns to drop'
                , d)
    df = df.drop(dropcolumns,axis=1)
    st.toast("Columns dropped succesfuly")
    st.write(df.head())
    match TaskType(df,target):
        case 'regression':
            expert = RegressionExpert()
        case 'classification':
            expert = ClassificationExpert()
    st.write(TaskType(df, target))
    HandleNumericNan = st.selectbox('Choose how to handle numeric null values', ["mean","mode","median"])
    HandleCategoricalNan = st.selectbox('Choose how to handle categorical null values', ["mode","add a new category"])
    if HandleCategoricalNan == "add a new category":
        HandleCategoricalNan = st.text_input("Enter the category")
    TestSize = st.slider('choose test data size', min_value=0.1, max_value=0.5)
    expert.setup(df, target, HandleNumericNan, HandleCategoricalNan,TestSize)
    st.write(expert.CompareModels())
    row = []
    rowdf = pd.DataFrame(index=[0,1])
    st.write(rowdf)
    st.write(f"please enter data to detect its {target}")
    for i in range(expert.x.shape[1]):
        if(expert.x.iloc[:,i].dtype.kind in 'OS'):
            row.append(st.text_input(f"Enter text {i}"))
        else:
            row.append(st.number_input(f"Enter number {i}"))
    for i in range(expert.x.shape[1]):
        rowdf.insert(i, expert.x.columns.tolist()[i], row[i], True)
    rowdf.drop(index=1,inplace=True)
    st.write(rowdf)
    rowdf = PreProcess(df=rowdf,numerical=HandleNumericNan
                    ,categorical=HandleCategoricalNan,target=None)
    st.write(expert.predict(rowdf))
    #st.write(expert.lnr.predict(df.drop(df[target],axis=1)))
# s = ClassificationExpert()
# s.setup(df, "Purchased")
# print(s.info)
# s.CompareModels()
# s.plot()













































