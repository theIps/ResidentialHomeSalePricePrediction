############################################################################
#Importing the packages being used
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb

#Random values generated should be same everytime the program is run
np.random.seed(7)

#Reading of Training and Testing Data from CSV File
data=pd.read_csv("train.csv")
train=data.iloc[0:1200,:]
test=data.iloc[1200:,:]
print(train.shape)
print("Number of Features in training dataset:-",train.shape[1])
print("Number of Features in testing dataset:-",test.shape[1])
#Dropping columns having all values as "NaN"
train=train.dropna(axis=1,how='all')
test=test.dropna(axis=1,how='all')

#Feature Extraction

#Adding the Total Square Feet of House and combining it into one variable
train['1stFlrSF']=train['1stFlrSF']+train['2ndFlrSF']
test['1stFlrSF']=test['1stFlrSF']+test['2ndFlrSF']

#Adding the Total Square Feet of Porch and combining it into one variable
train['OpenPorchSF']=train['OpenPorchSF']+train['EnclosedPorch']+train['3SsnPorch']+train['ScreenPorch']
test['OpenPorchSF']=test['OpenPorchSF']+test['EnclosedPorch']+test['3SsnPorch']+test['ScreenPorch']

#Adding the Total number of Baths in the House and combining it into one variable
train['FullBath']=train['FullBath']+train['BsmtHalfBath']+train['BsmtFullBath']+train['HalfBath']
test['FullBath']=test['FullBath']+test['BsmtHalfBath']+test['BsmtFullBath']+test['HalfBath']

#added a new feature i.e age of the house from the year built
train["Age"] = 2011 - train["YearBuilt"]
train["TimeSinceSold"] = 2011 - train["YrSold"]
#added a new feature i.e time since last the house was sold of the house from the year sold feature
test["Age"] = 2011 - test["YearBuilt"]
test["TimeSinceSold"] = 2011 - test["YrSold"]

#Dropping Unnecessary Features
train=train.drop(['Id','Alley','Fence','HalfBath','BsmtFinSF1','BsmtFullBath','BsmtHalfBath','Condition2','MoSold','BedroomAbvGr','ScreenPorch','3SsnPorch','EnclosedPorch','2ndFlrSF','LowQualFinSF','GarageCars','PoolQC','LotFrontage','MasVnrArea','BsmtFinSF2','BsmtFinType2','BsmtExposure','2ndFlrSF'],1)

test=test.drop(['Id','Alley','Fence','HalfBath','BsmtFinSF1','BsmtFullBath','BsmtHalfBath','Condition2','MoSold','BedroomAbvGr','ScreenPorch','3SsnPorch','EnclosedPorch','2ndFlrSF','LowQualFinSF','GarageCars','PoolQC','LotFrontage','MasVnrArea','BsmtFinSF2','BsmtFinType2','BsmtExposure','2ndFlrSF'],1)

#Feature Transformation
quality_dictionary = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

train["ExterCond"] = train["ExterCond"].map(quality_dictionary).astype(int)
train["BsmtQual"] = train["BsmtQual"].map(quality_dictionary).astype(int)
train["BsmtCond"] = train["BsmtCond"].map(quality_dictionary).astype(int)
train["KitchenQual"] = train["KitchenQual"].map(quality_dictionary).astype(int)
train["FireplaceQu"] = train["FireplaceQu"].map(quality_dictionary).astype(int)
train["GarageQual"] = train["GarageQual"].map(quality_dictionary).astype(int)
train["GarageCond"] = train["GarageCond"].map(quality_dictionary).astype(int)
train["ExterQual"] = train["ExterQual"].map(quality_dictionary).astype(int)
train["HeatingQC"] = train["HeatingQC"].map(quality_dictionary).astype(int)

test["ExterCond"] = test["ExterCond"].map(quality_dictionary).astype(int)
test["BsmtQual"] = test["BsmtQual"].map(quality_dictionary).astype(int)
test["BsmtCond"] = test["BsmtCond"].map(quality_dictionary).astype(int)
test["KitchenQual"] = test["KitchenQual"].map(quality_dictionary).astype(int)
test["FireplaceQu"] = test["FireplaceQu"].map(quality_dictionary).astype(int)
test["GarageQual"] = test["GarageQual"].map(quality_dictionary).astype(int)
test["GarageCond"] = test["GarageCond"].map(quality_dictionary).astype(int)
test["ExterQual"] = test["ExterQual"].map(quality_dictionary).astype(int)
test["HeatingQC"] = test["HeatingQC"].map(quality_dictionary).astype(int)

garag_dict = {None: 0, "Fin": 2, "RFn": 1, "Unf": 0}
train["GarageFinish"] = train["GarageFinish"].map(garag_dict).astype(int)
test["GarageFinish"] = test["GarageFinish"].map(garag_dict).astype(int)

house_dict = {None: 0, "1Story": 1,"1.5Unf":2,"1.5Fin":3, "SFoyer":3 ,"SLvl":3,"2Story": 4,"2.5Unf":5,"2.5Fin":6}
train["HouseStyle"] = train["HouseStyle"].map(house_dict).astype(int)
test["HouseStyle"] = test["HouseStyle"].map(house_dict).astype(int)


#Creating Neighborhood Map w.r.t below query i.e. giving each neighbourhood a ranking based on their sale price value
#train["SalePrice"].groupby(train["Neighborhood"]).median().sort_values()
neighborhood_map = {
        "MeadowV" : 0,
        "IDOTRR" : 1,"BrDale" : 1,"OldTown" : 1,"Edwards" : 1,"BrkSide" : 1,"Sawyer" : 1,"Blueste" : 1,
        "NPkVill": 2,"Mitchel": 2,"SWISU" : 2,"NAmes" : 2,"SawyerW" : 2,"Gilbert" : 2,"NWAmes" : 2,"Blmngtn" : 2,"CollgCr" : 2,
        "Crawfor" : 3,"Veenker" : 3,"ClearCr" : 3,"Somerst" : 3,"Timber" : 3,
        "StoneBr" : 4,"NridgHt" : 4,"NoRidge" : 4,
    }

train["Neighborhood"] = train["Neighborhood"].map(neighborhood_map)
test["Neighborhood"] = test["Neighborhood"].map(neighborhood_map)



#Selecting Categorical Features/Columns
categ_columns = train.select_dtypes(include=['object']).copy().columns
#Transforming Categorical Features
#Doing One Hot Encoding
for column in categ_columns:
        onehot_train = pd.get_dummies(train[column], prefix=column)
        train = train.drop(column, 1)
        train = train.join(onehot_train)
        onehot_test = pd.get_dummies(test[column], prefix=column)
        test = test.drop(column, 1)
        test = test.join(onehot_test)

print("Number of Features in training dataset(after categorical columns tranformation ):-",
      train.shape[0])
print("Number of Features in testing dataset(after categorical columns tranformation ):-",
      test.shape[0])

#Transforming Columns containg "NaN" Values
#train=train.dropna(axis=0,how="any")
for col in train.columns:
    val = train[col].min()
    train[col]=train[col].fillna(value=val,axis=0)

for col in test.columns:
    val = test[col].min()
    test[col]=test[col].fillna(value=val,axis=0)

#Separating training data Features and Predictions into different dataframes
X_train=train.drop('SalePrice',axis=1)
Y_train=train['SalePrice']

#Separating testing data Features and Predictions into different dataframes
X_test=test.drop('SalePrice',axis=1)
Y_test=test['SalePrice']

#Performing Feature Selection on Training Dataset
X_train_temp = X_train.copy(deep=True)  # Make a deep copy of the Training Data dataframe
selector = VarianceThreshold(0.12)
selector.fit(X_train_temp)
X_res = X_train_temp.loc[:, selector.get_support(indices=False)]
X_train=X_res
print("Number of Features in training dataset(after feature selection):-",X_train.shape[1])

#Selecting Same Features for testing dataset
X_test=X_test[X_train.columns]
print(X_test.shape[1])
print(X_test.columns)


#this is not used in this as I have used variance threshold for feature selection but this is also one of theways of doing feature selection
print(X_train.shape)
traini = pd.concat([X_train, Y_train], axis=1, join='inner')
# Find most important features relative to target i.e finding correlation of every individual feature i.e independent variable with dependent variable and then sorting them and using the features that have maximum correlation
print("Find most important features relative to target")
corr = traini.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)

#Splitting training dataset into train dataset and cross-validation dataset
X_train, X_CV, Y_train, Y_CV = train_test_split(X_train,Y_train,test_size=0.30)



#Applying Different Models to the dataset


#Fitting linear regression model
min=1000
min_norm=True
for norm in [False,True]:
    lin_reg = linear_model.LinearRegression(normalize=norm)
    lin_reg=lin_reg.fit(X_train,Y_train)
    #Predicting test dataset for linear regression model
    lin_pred=lin_reg.predict(X_CV)
    RMLSE = np.sqrt(mean_squared_log_error(Y_CV,lin_pred))
    #print("Error (Linear)=", RMLSE," for norm=",norm)
    if (min>RMLSE):
        min_norm=norm
        min=RMLSE

print("Root Mean Square Logarithmic Cross Validation Error (Linear)=",min,"Normalization=",min_norm)
lin_reg = linear_model.LinearRegression(normalize=min_norm)
lin_reg=lin_reg.fit(X_train,Y_train)
#Predicting test dataset for linear regression model
lin_pred=lin_reg.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(Y_test,lin_pred))
print ("Root Mean Square Logarithmic Generalize Error (Linear)=",RMLSE)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, lin_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["Linear Regression"])
plt.show()


#Fitting Lasso regression model
min=1000
for alp in [0.1,0.3,0.5,0.01,0.03,0.05]:
    for iter in [100,500,1000]:
        lass_reg = linear_model.Lasso(normalize=True,alpha=alp,max_iter=iter)
        lass_reg=lass_reg.fit(X_train,Y_train)
        #Predicting test dataset for lasso regression model
        lr_pred=lass_reg.predict(X_CV)
        RMLSE = np.sqrt(mean_squared_log_error(Y_CV,lr_pred))
        #print("Error (Lasso)=", RMLSE," for alpha=",alp," and max_iter=",iter)
        if (min > RMLSE):
            min = RMLSE
            min_alpha=alp
            min_iter=iter

print ("Root Mean Square Logarithmic Error Cross Validation(Lasso)=",min,"Alpha =",min_alpha,", max_iter=",min_iter)
lass_reg = linear_model.Lasso(normalize=min_norm,alpha=min_alpha,max_iter=min_iter)
lass_reg=lass_reg.fit(X_train,Y_train)
#Predicting test dataset for lasso regression model
lass_pred=lass_reg.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(Y_test,lass_pred))
print("Root Mean Square Logarithmic Generalize Error (Lasso)=", RMLSE)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, lass_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["Lasso Regression"])
plt.show()


#Fitting Elastic Net regression model
min=1000
for alp in [0.1,0.3,0.5,0.01,0.03,0.05]:
    for iter in [100,500,1000]:
        for norm in [True,False]:
            ela_reg = linear_model.ElasticNet(alpha=alp,max_iter=iter,normalize=norm)
            ela_reg=ela_reg.fit(X_train,Y_train)
            #Predicting test dataset for elastic net regression model
            el_pred=ela_reg.predict(X_CV)
            RMLSE = np.sqrt(mean_squared_log_error(Y_CV,el_pred))
            #print("Error (Elastic Net)=", RMLSE," for normalize=",norm ," for alpha=", alp,
            #      " and max_iter=", iter)
            if (min > RMLSE):
                min = RMLSE
                min_alpha = alp
                min_iter = iter
                min_norm=norm

print ("Root Mean Square Logarithmic Error Cross Validation(Elastic Net)=",min,"Alpha =",min_alpha,", max_iter=",min_iter," and normalize=",min_norm)
ela_reg = linear_model.ElasticNet(alpha=min_alpha,max_iter=min_iter,normalize=min_norm)
ela_reg=ela_reg.fit(X_train,Y_train)
#Predicting test dataset for elastic regression model
el_pred=ela_reg.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(Y_test,el_pred))
print ("Root Mean Square Logarithmic Generalize Error (Elastic Net)=",RMLSE)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, el_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["Elastic Net Regression"])
plt.show()



#Fitting Ridge regression model
min=1000
for alp in [0.1,0.3,0.5,0.01,0.03,0.05]:
    for iter in [100,500,1000]:
        for norm in [True,False]:
            ridg_reg = linear_model.Ridge(normalize=True)
            ridg_reg = ridg_reg.fit(X_train, Y_train)
            # Predicting test dataset for ridge regression model
            ridg_pred = ridg_reg.predict(X_CV)
            RMLSE = np.sqrt(mean_squared_log_error(Y_CV, ridg_pred))
            #print("Error (Ridge)=", RMLSE," for normalize=",norm ," for alpha=", alp,
            #      " and max_iter=", iter)
            if (min > RMLSE):
                min = RMLSE
                min_alpha = alp
                min_iter = iter
                min_norm=norm

print("Root Mean Square Logarithmic Error Cross Validation(Ridge)=", min, "Alpha =", min_alpha, ", max_iter=", min_iter, " and normalize=", min_norm)
ridg_reg = linear_model.Ridge(alpha=min_alpha,max_iter=min_iter,normalize=min_norm)
ridg_reg = ridg_reg.fit(X_train, Y_train)
# Predicting test dataset for ridge regression model
ridg_pred = ridg_reg.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(Y_test, ridg_pred))
print ("Root Mean Square Logarithmic Generalize Error (Ridge)=",RMLSE)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, ridg_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["Ridge Regression"])
plt.show()



#Fitting Random Forest
#Tuning Some of the Parameters for Random Forest
min=1000
for estimators in [5,10,12,14,15,16,18,20]:
    for depth in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        rforest=RandomForestRegressor(n_estimators=estimators, max_depth=depth, random_state=0)
        rforest=rforest.fit(X_train,Y_train)
        #Predicting test dataset
        forest_pred=rforest.predict(X_CV)
        RMLSE = np.sqrt(mean_squared_log_error(Y_CV, forest_pred))
        #print("Error (Forest)=", RMLSE, "for depth-", depth, "for estimators-", estimators)
        if (min>RMLSE):
            min=RMLSE
            dept=depth
            est=estimators
print ("Root Mean Square Logarithmic Cross Validation Error (Forest)=",min,"for depth-",dept,"for estimators-",est)
params = {'n_estimators': est, 'max_depth': dept, 'min_samples_split': 2}
rforest = rforest.fit(X_train, Y_train)
# Predicting test dataset
forest_pred = rforest.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(Y_test, forest_pred))
print ("Root Mean Square Logarithmic Generalize Error (Forest)=",RMLSE,"for depth-",dept,"for estimators-",est)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, forest_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["Random Forest Regression"])
plt.show()



#Fitting Gradient Boosting Regressor
#Tuning Some of the Parameters for Gradient Boosting Regressor
min=1000
for eta in [0.06,0.65,0.07,0.075,0.08]:
    for esti in [570,580,590,600,630,635,640]:
        for dep in [2]:
            params = {'n_estimators': esti, 'max_depth': dep, 'min_samples_split': 2,'learning_rate': eta, 'loss': 'ls'}
            clf = ensemble.GradientBoostingRegressor(**params)
            gboost=clf.fit(X_train,Y_train)
            #Predicting test dataset
            gb_pred=gboost.predict(X_CV)
            RMLSE = np.sqrt(mean_squared_log_error(Y_CV, gb_pred))
            #print("Error (Gradient Boosting)=", RMLSE, "for learning rate", eta," and for estimators=",esti)
            if (min>RMLSE):
                min = RMLSE
                etaf=eta
                estif=esti
                depf=dep

print ("Root Mean Square Logarithmic Cross Validation Error (Gradient Boosting)=",min,"for learning rate",etaf,"and for estimators"
       ,estif)
params = {'n_estimators': estif, 'max_depth': depf, 'min_samples_split': 2,'learning_rate': etaf, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
gboost=clf.fit(X_train,Y_train)
#Predicting test dataset
gb_pred=gboost.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(Y_test, gb_pred))
print("Root Mean Square Logarithmic Generalize Error (Gradient Boosting)=", RMLSE, "for learning rate", etaf," and for estimators=",estif)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, gb_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["Gradient Boosting Regression"])
plt.show()



#Fitting XGBoost
min=1000
for lr in [0.01,0.03,0.05,0.1,0.3]:
    regr = xgb.XGBRegressor(
                colsample_bytree=0.2,
                gamma=0.0,
                learning_rate=lr,
                max_depth=4,
                min_child_weight=2,
                n_estimators=7200,
                reg_alpha=0.9,
                reg_lambda=0.6,
                subsample=0.2,
                seed=42,
                silent=True)


    regr.fit(X_train, Y_train)
    xg_pred = regr.predict(X_CV)
    RMLSE = np.sqrt(mean_squared_log_error(Y_CV, xg_pred))
    #print ("Error (XG Boost)=",RMLSE," for learning rate=",lr)
    if (min>RMLSE):
        min = RMLSE
        lr_f=lr

print("Root Mean Square Logarithmic Cross Validation Error (XG Boosting)=", min, "for learning rate", lr_f)
regr = xgb.XGBRegressor(
                colsample_bytree=0.2,
                gamma=0.0,
                learning_rate=lr_f,
                max_depth=4,
                min_child_weight=2,
                n_estimators=7200,
                reg_alpha=0.9,
                reg_lambda=0.6,
                subsample=0.2,
                seed=42,
                silent=True)
regr.fit(X_train, Y_train)
xg_pred = regr.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(Y_test, xg_pred))
print ("Root Mean Square Logarithmic Generalize Error (XG Boost)=",RMLSE)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, xg_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["XG Boost Regression"])
plt.show()



#Fitting AdaBoostRegressor
min=1000
for dep in [10,15,18,25]:
    for esti in [550,575,600]:
        for lr in [0.01,0.3,1.25,1.5]:
            regr_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=dep),n_estimators=esti,random_state=0,
                             learning_rate=lr,loss="exponential")
            regr_ada.fit(X_train, Y_train)
            ada_pred=regr_ada.predict(X_CV)
            RMLSE = np.sqrt(mean_squared_log_error(Y_CV, ada_pred))
            #print ("Error (AdaBoostRegressor)=",RMLSE," for depth=",dep," for estimators=",esti," and learning rate=",lr)
            if (min > RMLSE):
                min = RMLSE
                lr_f=lr
                esti_f=esti
                dep_f=dep

print ("Root Mean Square Logarithmic Cross Validation Error (AdaBoostRegressor)=",RMLSE," for depth=",dep,
       " for estimators=",esti," and learning rate=",lr)
regr_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=dep_f), n_estimators=esti_f, random_state=0,
                                         learning_rate=lr_f, loss="exponential")
regr_ada.fit(X_train, Y_train)
ada_pred = regr_ada.predict(X_test)
RMLSE = np.sqrt(mean_squared_log_error(Y_test, ada_pred))
print("Error (AdaBoostRegressor)=", RMLSE, " for depth=", dep, " for estimators=", esti," and learning rate=", lr)
print ("Root Mean Square Logarithmic Generalize Error (AdaBoostRegressor)=",RMLSE,
       " for depth=",dep_f," for estimators=",esti_f," and learning rate=",lr_f)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, ada_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["AdaBoost Regression"])
plt.show()



#Blending Models
blend_pred=(ada_pred+gb_pred+xg_pred+forest_pred)/4
RMLSE = np.sqrt(mean_squared_log_error(Y_test, blend_pred))
print ("Root Mean Square Logarithmic Error (Blending Models)=",RMLSE)

#Plotting
fig, ax = plt.subplots()
ax.scatter(Y_test, blend_pred, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.legend(["Models Blended"])
plt.show()


