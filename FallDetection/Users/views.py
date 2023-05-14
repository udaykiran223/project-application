from django.shortcuts import render
import sqlite3
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import time
import webbrowser
from sklearn.ensemble import RandomForestClassifier

# Create your views here.
def index(request):
    return render(request,'Users/index.html')
def login(request):
    return render(request,'Users/Users.html')
def Register(request):
    return render(request,'Users/Register.html')
def RegAction(request):
    name=request.POST['name']
    email=request.POST['email']
    mobile=request.POST['mobile']
    address=request.POST['address']
    username=request.POST['username']
    password=request.POST['password']
    conn = sqlite3.connect('falldetection.db')
    cur=conn.cursor()
    #conn.execute("create table user(name varchar(100),email varchar(100),mobile varchar(100),address varchar(100),username varchar(100),password varchar(100))")
    cur.execute("select * from user where username='"+username+"'")
    dd=cur.fetchone()
    if dd is not None:
        context={'data':'This User Already Exist..!!'}
        return render(request,'Users/Users.html',context)
    else:
        cur.execute("insert into user values('"+name+"','"+email+"','"+mobile+"','"+address+"','"+username+"','"+password+"')")
        conn.commit()
        context={'data':'Registration Successful..!!'}
        return render(request,'Users/Users.html', context)
    
def LogAction(request):
    username=request.POST.get('username')
    password=request.POST.get('password')
    con=sqlite3.connect('falldetection.db')
    cur=con.cursor()
    cur.execute("select * from user where username='"+username+"'and password='"+password+"'")
    data=cur.fetchone()
    if data is not None:
        request.session['username']=username
        return render(request,'Users/UserHome.html')
    else:
        context={'data':'Login Failed ....!!'}
        return render(request,'Users/Users.html',context)
def home(request):
    return render(request,'Users/AdminHome.html')
global df
global dataset
def LoadData(request):
    global dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset=pd.read_csv(BASE_DIR+"\\dataset\\falldeteciton.csv")
    #data.fillna(0, inplace=True)
    context={'data':"Dataset Loaded\n"}
    
    return render(request,'Users/UserHome.html',context)

global X, Y
global svm_accuracy, dt_accuracy, svm_train_time, dt_train_time,svm_test_time, dt_test_time, rf_accuracy, rf_train_time, rf_test_time
global X_train, X_test, y_train, y_test
global scaler
global classifier

label = ['Standing','Walking','Sitting','Falling','Cramps','Running']

def preprocess(request):
    global dataset
    #label = dataset.groupby('ACTIVITY').size()#ploting graph with number of on time and default payment with class label as 0 and 1
    #label.plot(kind="bar")
    #plt.show()

    global dataset
    global X, Y
    global X_train, X_test, y_train, y_test
    global scaler
    dataset.fillna(0, inplace = True) #replacing missing or NA values with 0 in dataset
    dataset = dataset.values #converting entire dataset into values and assign to X
    Y = dataset[:,0]
    X = dataset[:,1:dataset.shape[1]]
    scaler = MinMaxScaler() 
    scaler.fit(X) #applying MIX-MAX function on dataset to preprocess dataset
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    context={"total":len(dataset),"Training":len(X_train),"Testing":len(X_test)}
    return render(request,'Users/Preprocess.html',context)


def runSVMAlgorithm(request):
    global X, Y
    global svm_accuracy, svm_train_time, svm_test_time
    global X_train, X_test, y_train, y_test
    start = time.time()
    cls = svm.SVC()
    cls.fit(X,Y)
    end = time.time()
    svm_train_time = end - start
    start = time.time()
    svm_predict = cls.predict(X_test)
    svm_accuracy = accuracy_score(svm_predict, y_test)*100
    ad = format(svm_accuracy, ".2f")
    end = time.time()
    svm_test_time = end - start
    context={"Accurary":str(svm_accuracy),"traintime":str(svm_train_time),"testtime":str(svm_test_time)}
    return render(request,'Users/SVMPage.html',context)

    

def runDecistionTree(request):
    global classifier
    global X, Y
    global dt_accuracy, dt_train_time, dt_test_time
    global X_train, X_test, y_train, y_test
    start = time.time()
    classifier = DecisionTreeClassifier()
    classifier.fit(X, Y)
    end = time.time()
    dt_train_time = end - start
    start = time.time()
    svm_predict = classifier.predict(X_test)
    dt_accuracy = accuracy_score(svm_predict, y_test)*100
    end = time.time()
    dt_test_time = end - start
    context={"Accurary":str(dt_accuracy),"traintime":str(dt_train_time),"testtime":str(dt_test_time)}
    return render(request,'Users/DTPage.html',context)
def runRandomForest(request):
    global X, Y
    global rf_accuracy, rf_train_time, rf_test_time
    global X_train, X_test, y_train, y_test
    start = time.time()
    classifier = RandomForestClassifier()
    classifier.fit(X, Y)
    end = time.time()
    rf_train_time = end - start
    start = time.time()
    svm_predict = classifier.predict(X_test)
    rf_accuracy = accuracy_score(svm_predict, y_test)*100
    end = time.time()
    rf_test_time = end - start
    context={"Accurary":str(rf_accuracy),"traintime":str(rf_train_time),"testtime":str(rf_test_time)}
    return render(request,'Users/RandomForestPage.html',context)
    
  
def runComparision(request):   
    global svm_accuracy,dt_accuracy,rf_accuracy
    height = [svm_accuracy,dt_accuracy, rf_accuracy]
    bars = ('SVM Accuracy','Decision Tree Accuracy','Random Forest Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    return render(request,'Users/UserHome.html')

def predict(request):
    global scaler
    global classifier
    list = []
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset=pd.read_csv(BASE_DIR+"\\dataset\\testData.csv")
    dataset = dataset.values 
    temp = dataset
    test = dataset[:,0:dataset.shape[1]]
    test = scaler.transform(test)
    print(test.shape)
    predict = classifier.predict(test)
    print(predict)
    for i in range(len(temp)):
        list.append("Test Record = "+str(temp[i])+" PREDICTION = "+label[int(predict[i])]+" ||")
    print(list)
    context={'data':list}
    return render(request,'Users/Prediction.html',context)

        
    
        
        
    
    



    




    

