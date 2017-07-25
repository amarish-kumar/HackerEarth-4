import pandas as pd
import numpy as np

df_train=pd.read_csv("train.csv")
#print(df_train)
#print(df_train.isnull().sum())
#Percent of NaN Values for Training Data:
#siteid: 9.98%
#browserid: 5%
#devid: 15%
df_test=pd.read_csv("test.csv")
#print(df_test)
#print(df_test.isnull().sum())
#Percent of NaN Values for Testing Data:
#siteid: 10%
#browserid: 5.98%
#devid: 19%
#As every ad has a unique id,drop the id column.
id_test=df_test.as_matrix(columns=["ID"]).ravel()
df_train.drop(["ID"],axis=1,inplace=True)
print("TRAINING DATA DESCRIPTION")
print(df_train["siteid"].describe())
print(df_train["browserid"].describe())
print(df_train["devid"].describe())


print("TESTING DATA DESCRIPTION")
print(df_test["siteid"].describe())
print(df_test["browserid"].describe())
print(df_test["devid"].describe())

# Filling NaN values
df_train["siteid"].fillna(5023971,inplace=True)    #Filled the mean value
df_train["browserid"].fillna("Edge",inplace=True)  # Filled the top value
df_train["devid"].fillna("Mobile",inplace=True)    # Filled the top value

df_test["siteid"].fillna(5046957,inplace=True)      #Filled the mean value
df_test["browserid"].fillna("Edge",inplace=True)    # Filled the top value
df_test["devid"].fillna("Mobile",inplace=True)      # Filled the top value


df_test.drop(["ID"],axis=1,inplace=True)
#print(df_train.isnull().sum())
#print(df_test.isnull().sum())
labels_train=df_train.as_matrix(columns=["click"]).ravel()
print(labels_train)
df_train.drop(["click"],axis=1,inplace=True) #Dropping "click" from features_train

from sklearn.preprocessing import LabelEncoder
#le=LabelEncoder()
df_train["countrycode"]=LabelEncoder().fit_transform(df_train["countrycode"])
df_test["countrycode"]=LabelEncoder().fit_transform(df_test["countrycode"])

df_train["browserid"]=LabelEncoder().fit_transform(df_train["browserid"])
df_test["browserid"]=LabelEncoder().fit_transform(df_test["browserid"])

df_train["devid"]=LabelEncoder().fit_transform(df_train["devid"])
df_test["devid"]=LabelEncoder().fit_transform(df_test["devid"])

######################## TESTING PURPOSES ONLY ########################################
#for i in range(len(features_train)):
#    if not("a" in features_train[i][4] or "b" in features_train[i][4] or "c"  in features_train[i][4] or "d"  in features_train[i][4] or "e"  in features_train[i][4] or "f"  in features_train[i][4]):
#        print(features_train[i][4])
#print("##################################")

#for i in range(len(features_test)):
#    if not("a" in features_test[i][4] or "b" in features_test[i][4] or "c"  in features_test[i][4] or "d" in features_test[i][4] or "e" in features_test[i][4] or "f" in features_test[i][4]):
#        print(features_test[i][4])
########################################################################################
# Since all the action takes place in January (10th to 23rd),I will be using absolute time as a feature.
# Also day_of_week can be used as a feature.
# Codes for:
# Abs_time: 24 hour format
# day_of_week: 1: Sunday 2: Monday 3: Tuesday 4: Wed 5: Thurs 6: Fri 7: Sat

datetime_tr=df_train.as_matrix(columns=["datetime"]).ravel()
siteid_tr=df_train.as_matrix(columns=["siteid"]).ravel()
offerid_tr=df_train.as_matrix(columns=["offerid"]).ravel()
category_tr=df_train.as_matrix(columns=["category"]).ravel()
merchant_tr=df_train.as_matrix(columns=["merchant"]).ravel()
countrycode_tr=df_train.as_matrix(columns=["countrycode"]).ravel()
browserid_tr=df_train.as_matrix(columns=["browserid"]).ravel()
devid_tr=df_train.as_matrix(columns=["devid"]).ravel()



datetime_te=df_test.as_matrix(columns=["datetime"]).ravel()
siteid_te=df_test.as_matrix(columns=["siteid"]).ravel()
offerid_te=df_test.as_matrix(columns=["offerid"]).ravel()
category_te=df_test.as_matrix(columns=["category"]).ravel()
merchant_te=df_test.as_matrix(columns=["merchant"]).ravel()
countrycode_te=df_test.as_matrix(columns=["countrycode"]).ravel()
browserid_te=df_test.as_matrix(columns=["browserid"]).ravel()
devid_te=df_test.as_matrix(columns=["devid"]).ravel()


abs_time_tr=[]
day_of_week_tr=[]
# New Features
hour_of_day_tr=[]
minute_of_hour_tr=[]
second_of_minute_tr=[]

for s in datetime_tr:
    date,time=s.split()
    date=date[-2:]
    time=time.replace(":","")
    if date=="10" or date=="17":
        day_of_week_tr.append(3)
    elif date=="11" or date=="18":
        day_of_week_tr.append(4)
    elif date=="12" or date=="19":
        day_of_week_tr.append(5)
    elif date=="13" or date=="20":
        day_of_week_tr.append(6)
    elif date=="14" or date=="21":
        day_of_week_tr.append(7)
    elif date=="15" or date=="22":
        day_of_week_tr.append(1)
    elif date=="16" or date=="23":
        day_of_week_tr.append(2)
    abs_time_tr.append(int(time))
    hr=time[:2]
    minute=time[2:4]
    second=time[4:]
    hour_of_day_tr.append(int(hr))
    minute_of_hour_tr.append(int(minute))
    second_of_minute_tr.append(int(second))





abs_time_te=[]
day_of_week_te=[]
hour_of_day_te=[]
minute_of_hour_te=[]
second_of_minute_te=[]



for s in datetime_te:
    date,time=s.split()
    date=date[-2:]
    time=time.replace(":","")
    if date=="10" or date=="17":
        day_of_week_te.append(3)
    elif date=="11" or date=="18":
        day_of_week_te.append(4)
    elif date=="12" or date=="19":
        day_of_week_te.append(5)
    elif date=="13" or date=="20":
        day_of_week_te.append(6)
    elif date=="14" or date=="21":
        day_of_week_te.append(7)
    elif date=="15" or date=="22":
        day_of_week_te.append(1)
    elif date=="16" or date=="23":
        day_of_week_te.append(2)
    abs_time_te.append(int(time))
    hr = time[:2]
    minute = time[2:4]
    second = time[4:]
    hour_of_day_te.append(int(hr))
    minute_of_hour_te.append(int(minute))
    second_of_minute_te.append(int(second))

features_train=[]

for i in range(len(offerid_tr)):
    features_train.append([day_of_week_tr[i],hour_of_day_tr[i],minute_of_hour_tr[i],second_of_minute_tr[i],abs_time_tr[i],siteid_tr[i],offerid_tr[i],category_tr[i],merchant_tr[i],countrycode_tr[i],browserid_tr[i],devid_tr[i]])

features_train = np.asarray(features_train)

features_test=[]

for i in range(len(offerid_te)):
    features_test.append([day_of_week_te[i],hour_of_day_te[i],minute_of_hour_te[i],second_of_minute_te[i],abs_time_te[i],siteid_te[i],offerid_te[i],category_te[i],merchant_te[i],countrycode_te[i],browserid_te[i],devid_te[i]])

features_test = np.asarray(features_test)

features_train=features_train.astype(np.float)
features_test=features_test.astype(np.float)

from sklearn.preprocessing import MinMaxScaler as mms
scalar=mms()
features_train=scalar.fit_transform(features_train)
features_test=scalar.fit_transform(features_test)
print(features_train)
print(features_test)


def random_forest(f_train,l_train,f_test):
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.grid_search import GridSearchCV
    #param={'criterion' : ('gini','entropy'),'min_samples_split':[2,5,10,15,20,25,30],'n_estimators':[100]}
    #svr=RandomForestClassifier()
    #clf=GridSearchCV(svr,param)
    clf=RandomForestClassifier()
    import time
    start_time=time.time()
    clf.fit(f_train,l_train)
    print("Training Time: %s seconds" % (time.time() - start_time))
    #print(clf.best_params_)
    start_time = time.time()
    pred=clf.predict_proba(f_test)
    print("Predicting Time: %s seconds" % (time.time() - start_time))
    print(pred)
    return pred

def gradient_boosting_classifier(f_train,l_train,f_test):
    from sklearn.ensemble import GradientBoostingClassifier as gbc
    clf=gbc()
    import time
    start_time=time.time()
    clf.fit(f_train,l_train)
    print("Training Time: %s seconds"%(time.time() - start_time))
    start_time=time.time()
    pred=clf.predict_proba(f_test)
    print("Predicting Time: %s seconds" %(time.time() - start_time))
    return pred

def xg_boost(f_train,l_train,f_test):
    from xgboost import XGBClassifier as xgb
    clf=xgb(n_estimators=100)
    clf.fit(f_train,l_train)
    pred=clf.predict_proba(f_test)
    #print(pred)
    return pred

def writer(pred,id_te):
    txt = []

    for i in range(len(pred)):
        txt.append([str(id_te[i]), str(pred[i][1])])

    df_result = pd.DataFrame(txt)
    df_result.columns = ["ID", "click"]
    df_result.to_csv(path_or_buf="output_classifier.csv", index=False)


pred=xg_boost(features_train,labels_train,features_test)
writer(pred,id_test)






