import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import statistics as stats
import matplotlib.pyplot as plt
from scipy import stats
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

df = pd.read_csv(r"C:\Users\serha\OneDrive\Masaüstü\Bitirme\bitirmeSun/train.csv")

df.head()
df.describe()
df.isnull().sum()
print(df.iloc[:,40:].isnull().sum())
df.info()

plt.figure(figsize=(7,5))
sns.distplot(df['LotFrontage'])
plt.show()

df['Alley'].value_counts().plot(kind = 'pie',)
df['MasVnrType'].value_counts()

plt.figure(figsize=(14,6))
sns.displot(df['MasVnrArea'])
plt.show()

df['BsmtCond'].value_counts()
df['BsmtFinType1'].value_counts()
df['BsmtExposure'].value_counts()
df['GarageType'].value_counts()
df['GarageFinish'].value_counts()
df['GarageQual'].value_counts()
df['GarageCond'].value_counts()
df['LotFrontage'] = df['LotFrontage'].fillna(np.mean(df['LotFrontage']))
df['BsmtCond'] = df['BsmtCond'].fillna('TA')
df['BsmtQual'] = df['BsmtQual'].fillna('TA')
df['ExterQual'] = df['ExterQual'].fillna('No')
df['BsmtExposure'] = df['BsmtExposure'].fillna('No')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('Unf')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('Unf')
df['GarageType'] = df['GarageType'].fillna('Attchd')
df['GarageFinish'] = df['GarageFinish'].fillna('Unf')
df['GarageQual'] = df['GarageQual'].fillna('TA')
df['GarageCond'] = df['GarageCond'].fillna('TA')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(np.mean(df['MasVnrArea']))

df.drop(columns=['Alley','FireplaceQu','MiscFeature','PoolQC','Fence','MiscVal','Id'], axis=1,inplace=True)

df.dropna(inplace=True)
df.isnull().sum()
df.head()
len(df.columns)
df.corr()

#Dropping those values which giving negative correlation
df2 = df.drop(columns=['OverallCond','MSSubClass','BsmtFinSF2','LowQualFinSF','BsmtHalfBath',
                        'KitchenAbvGr','EnclosedPorch','YrSold'],axis=1)
df2.head()

df.drop(columns=['MSSubClass','OverallCond','YrSold','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','KitchenAbvGr',
                'EnclosedPorch','MoSold','ScreenPorch'], axis=1,inplace=True)
len(df.columns)

df.drop(columns=['3SsnPorch','PoolArea','BedroomAbvGr'],axis=1,inplace=True)
df.columns

df2.shape

print(df2.select_dtypes(include=[float]).columns)
df2['LotFrontage'] = df2['LotFrontage'].astype(int)
df2['MasVnrArea'] = df2['MasVnrArea'].astype(int)
df2['GarageYrBlt'] = df2['GarageYrBlt'].astype(int)

print(df2.select_dtypes(include=[float]).columns)
#Now No float columns in the  data

def Numerical(x,fig):
    plt.subplot(5,2,fig)
    plt.title(x+' Histogram')
    sns.displot(df2[x])
    plt.subplot(5,2,(fig+1))
    plt.title(x+' ScatterPLot')
    sns.scatterplot(x= df2[x],y=df2.SalePrice)
    
plt.figure(figsize=(25,30))
plt.tight_layout()
plt.show()

df_num = df2.select_dtypes(include=[int])
df_num.columns

plt.figure(figsize=(15,25))
Numerical('LotFrontage',1)
Numerical('LotArea',3)
Numerical('OverallQual',5)
Numerical('YearBuilt',7)
Numerical('YearRemodAdd',9)
plt.show()

plt.figure(figsize=(15,25))
Numerical('MasVnrArea',1)
Numerical('BsmtFinSF1',3)
Numerical('BsmtUnfSF',5)
Numerical('TotalBsmtSF',7)
Numerical('1stFlrSF',9)

plt.figure(figsize=(15,25))
Numerical('2ndFlrSF',1)
Numerical('GrLivArea',3)
Numerical('BsmtFullBath',5)
Numerical('FullBath',7)
Numerical('HalfBath',9)

plt.figure(figsize=(15,25))
Numerical('BedroomAbvGr',1)
Numerical('TotRmsAbvGrd',3)
Numerical('Fireplaces',5)
Numerical('GarageYrBlt',7)
Numerical('GarageCars',9)

df2.columns

plt.figure(figsize=(15,25))
Numerical('GarageArea',1)
Numerical('WoodDeckSF',3)
Numerical('OpenPorchSF',5)
Numerical('ScreenPorch',7)
Numerical('PoolArea',9)

warnings.filterwarnings('ignore')
df_train = pd.read_csv(r"C:\Users\serha\OneDrive\Masaüstü\Bitirme\bitirmeSun/train.csv")

grp_neighbrhd=df_train.groupby("Neighborhood")

unique_neighbrhd=df_train.Neighborhood.unique().tolist()

neighbrhd_list, SalePrice_list = [],[]
for item in unique_neighbrhd:
    df = grp_neighbrhd.get_group(item)
    neighbrhd_list.append(item)
    SalePrice_list.append(df.SalePrice.mean())

df_neighbrhd=pd.DataFrame({"Neighborhood": neighbrhd_list,
                          "SalePrice": SalePrice_list})
df_neighbrhd.sort_values("SalePrice",ascending=False,inplace=True)

sns.barplot(data=df_neighbrhd,x="Neighborhood",y="SalePrice");
plt.xticks(rotation=90);
plt.ylabel("Avg SalePrice");
plt.title("Average SalePrice based on Neighborhood");

del([grp_neighbrhd,df_neighbrhd,df,unique_neighbrhd])

#2. OverAll quality of the house and SalePrice

grp_overallqual=df_train.groupby("OverallQual")
unique_overallqual=df_train.OverallQual.unique()
sale_price_list,overall_qual=[],[]
for item in unique_overallqual:
    df=grp_overallqual.get_group(item)
    overall_qual.append(item)
    sale_price_list.append(df.SalePrice.mean())
df_overallqual=pd.DataFrame({"OverallQual":overall_qual,
                            "SalePrice": sale_price_list})

sns.lineplot(data=df_overallqual,x="OverallQual",y="SalePrice");
plt.title("SalePrice based on OverAllQuality");
plt.ylabel("Average SalePrice");
plt.xlabel("OverAll Quality");

del(df,df_overallqual)

#3. Age of the house and SalePrice
df=df_train.loc[:,["YearBuilt","YrSold","SalePrice"]].copy()
df["Age"]=df.YrSold-df.YearBuilt
print ("House age statistics:\n")
df.Age.describe()

grp_age=df.groupby("Age")
unique_age=df.Age.unique()
age_list,sale_price_list,count_list=[],[],[]
for age in unique_age:
    df_sale=grp_age.get_group(age)
    age_list.append(age)
    sale_price_list.append(df_sale.SalePrice.mean())
    count_list.append(df_sale.Age.count())
    
    df_sale=pd.DataFrame({"Age": age_list,
                     "SalePrice": sale_price_list,
                     "Count":count_list})
    
    sns.lineplot(data=df_sale,x="Age",y="SalePrice");
plt.title("Age of the house vs. SalePrice");
plt.ylabel("Average SalePrice");

sns.lineplot(data=df_sale,x="Age",y="Count");
plt.xticks(rotation=90);
plt.title("Count of reords");

df_input=df_train.copy()


#Belirli sütunlar için eksik değerleri işleme fonksiyonu
#Eksik değerler "NA" ile doldurulur ve bunu yaparken kullanıcıya hangi sütunlar için işlem yaptığını bildirilir.
#Fonksiyon sonunda, veri kümesinde eksik değerlerin var olup olmadığı kontrol edilir ve veri kümesi geri döndürülür.
#fonksiyonun sonunda, "SalePrice" sütununun eğrilik değeri hesaplanır ve ekrana yazdırılır. 

def process_imputation(df_input):
    df_input["PoolQC"].fillna("NA",inplace=True)
    print ("PoolQC - replaced missing values with NA")
    df_input.MiscFeature.fillna("NA",inplace=True)
    print ("MiscFeature - replaced missing values with NA")
    df_input.Alley.fillna("NA",inplace=True)
    print ("Alley - replaced missing values with NA")
    df_input.Fence.fillna('NA',inplace=True)
    print ("Fence - replaced missing values with NA")
    df_input.FireplaceQu.fillna("NA",inplace=True)
    print ("FireplaceQuality - replaced missing values with NA")
    df_input.LotFrontage.fillna(0,inplace=True)
    print ("Lot frontage - replaced missing values with zero")
    df_input.GarageType.fillna("NA",inplace=True)
    print ("Garage type - replaced missing values with NA")
    print ("GarageYrBlt - Replacing missing value with House built year")
    df_input.GarageYrBlt.fillna(df_input.YearBuilt,inplace=True)
    print ("GarageFinish - Replacing missing values with NA")
    df_input.GarageFinish.fillna('NA',inplace=True)
    print ("GarageQual - Replacing missing values with NA")
    df_input.GarageQual.fillna('NA',inplace=True)
    print ("GarageCond - Replacing missing values with NA")
    df_input.GarageCond.fillna('NA',inplace=True)
    for col in ["BsmtExposure","BsmtFinType2","BsmtFinType1","BsmtCond","BsmtQual"]:
        df_input[col].fillna("NA", inplace=True)
    print (f"{col} - replaced missing values with NA")
    df_input.MasVnrArea.fillna(0,inplace=True)
    print ("MasVnrArea - replaced missing values with 0")
    df_input.MasVnrType.fillna("None",inplace=True)
    print ("MasVnrType - replaced missing values with None")
    df_input.Electrical.fillna("NA",inplace=True)
    print ("Electrical - replaced missing values with NA")
    print ("Is there any missing values? ")
    print (df_input.isnull().any().value_counts().index)
    return df_input

df_input=process_imputation(df_input)

print (f"Target variable SalepPrice skewness: {df_input.SalePrice.skew()}")

sns.boxplot(data=df_input,x="SalePrice")
plt.title("SalePrice -original distribution");

total = df_input.shape[0]


plt.subplot(2,2,1)
sns.boxplot(data=df_input,x="SalePrice")
plt.suptitle("SalePrice -Ater removing outliers distribution");
plt.subplot(2,2,2)
plt.hist(data=df_input,x="SalePrice");
print ("Skewness",df_input.SalePrice.skew())

print ("Applying SquareRoot method of Saleprice for reducing the skewness.")
df_input["SalePrice_sqrt"] =np.sqrt(df_input.SalePrice)
plt.hist(df_input.SalePrice_sqrt);
plt.title("Square root of SalePrice - distribution");
print ("Skewness: ", df_input.SalePrice_sqrt.skew());


#Veri kümesinin şekli ekrana yazdırılır ve "Age" adlı yeni bir özellik oluşturulur, bu özellik "YrSold" ve "YearBuilt"
#sütunlarından hesaplanır. Daha sonra, "YearBuilt" ve "YrSold" sütunları veri kümesinden kaldırılır.
#Eksik değerleri yüksek olan "PoolQC", "MiscFeature" ve "Alley" gibi sütunlar veri kümesinden kaldırılır. 
#"Street", "Utilities", "Condition2", "RoofMatl", "Electrical", "Heating", "MoSold" ve "Id" gibi sütunlar veri kümesinden kaldırılır.
#"BathRooms" adlı yeni bir sütun oluşturulur ve "BsmtFullBath", "BsmtHalfBath", "FullBath" ve "HalfBath" sütunlarından hesaplanır 
#Bu sütunlar veri kümesinden kaldırılır. Ardından, "BsmtFinType1", "BsmtFinType2", "ExterQual", "ExterCond", "BsmtCond",
#"HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "BsmtExposure", "GarageFinish", "PavedDrive" ve "Fence"
#gibi sütunlar için kategorik değerler numerik değerlere dönüştürülür. Daha sonra "FlrSF" adlı yeni bir sütun oluşturulur ve
#"1stFlrSF" ve "2ndFlrSF" sütunlarından hesaplanır ve bu sütunlar veri kümesinden kaldırılır.
#Son olarak, veri kümesi "pd.get_dummies()" fonksiyonu kullanılarak one-hot encode edilir ve son hali geri döndürülür.
def wrangle_data(df):
    print (f"Shape of the dataframe before wrangling: {df.shape}")
    #Create new feature Age from year built and year sold
    df["Age"]=df["YrSold"] - df["YearBuilt"]
    print ("Created new feature 'Age' using Year sold and Year built")
    df.drop(["YearBuilt","YrSold"],axis=1,inplace=True)
    print ("Removed features - YearBuilt,YrSold")
    
    #Remove features having more missing values
    df.drop(["PoolQC","MiscFeature","Alley"],axis=1,inplace=True)
        
    #Below features contains meaningless value or presence of one value dominant.
    #Hence This features doesn't make any sense.So removing from dataset.
    del_vars=["Street","Utilities","Condition2","RoofMatl","Electrical","Heating", "MoSold","Id"]
    df.drop(del_vars,inplace=True,axis=1)
    
    df["BathRooms"]=df["BsmtFullBath"]+df["BsmtHalfBath"]+df["FullBath"]+df["HalfBath"]
    df.drop(["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"],axis=1,inplace=True)
            
    for var in ["BsmtFinType1","BsmtFinType2"]:
        df[var].replace(["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"],[6,5,4,3,2,1,0],inplace=True)
        
    for var in ["ExterQual","ExterCond","BsmtCond","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond"]:
        df[var].replace(["Ex","Gd","TA","Fa","Po","NA"],[5,4,3,2,1,0],inplace=True)
        
    for var in ["BsmtExposure"]:
        df[var].replace(["Gd","Av","Mn","No","NA"],[4,3,2,1,0],inplace=True)
        
    for var in ["GarageFinish"]:
        df[var].replace(["Fin","RFn","Unf","NA"],[3,2,1,0],inplace=True)
        
    for var in ["PavedDrive"]:
        df[var].replace(["Y","P","N"],[2,1,0],inplace=True)
    
    for var in ["Fence"]:
         df[var].replace(["GdPrv","MnPrv","GdWo","MnWw","NA"],[4,3,2,1,0],inplace=True)   
        
    df["BsmtQual"].replace(["Ex","Gd","TA","Fa","Po","NA"],[5,4,3,2,1,0],inplace=True)
        
    #Creating new feature FlrSf
    df["FlrSF"]=df["1stFlrSF"] + df["2ndFlrSF"]
    df.drop(["1stFlrSF","2ndFlrSF"],axis=1,inplace=True)
            
    df=pd.get_dummies(df)
    print (f"Shape of the dataframe after wrangling {df.shape}")
    return df

df_input=wrangle_data(df_input)

df_input.drop("SalePrice",axis=1,inplace=True)
print ("Removed the feature SalePrice")

#Checking for any String fields.
print("List of any string data?")
df_input.dtypes[df_input.dtypes=="object"]

corr_matrix=df_input.corr()

corr_matrix=pd.DataFrame(corr_matrix['SalePrice_sqrt']).sort_values('SalePrice_sqrt',ascending=False)

negative_corr_flds=corr_matrix[corr_matrix["SalePrice_sqrt"]<= 0].index.tolist()

#Remove negative correlated features
df_input.drop(negative_corr_flds,axis=1,inplace=True)

#Removing below derived features from dataset, since they are not present in the final test file.
df_input.drop(["HouseStyle_2.5Fin","Exterior1st_ImStucc","Exterior1st_Stone","Exterior2nd_Other"],axis=1,inplace=True)

corr_matrix_after=df_input.corr()
corr_matrix_after=pd.DataFrame(corr_matrix_after['SalePrice_sqrt']).sort_values('SalePrice_sqrt',ascending=False)
plt.figure(figsize=(3,15));
sns.heatmap(corr_matrix_after,cmap='inferno');

#Create independent and dependent features for model training.
x=df_input.drop('SalePrice_sqrt',axis=1)
y=df_input.SalePrice_sqrt
training_features=x.columns.tolist()

#Training and Testing split.
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
print (f"Shape of x train: {x_train.shape}, y train: {y_train.shape},x test: {x_test.shape}, y test: {y_test.shape}")

x.columns

plt.figure(figsize=(20,10));
plt.subplot(2,2,1)
plt.hist(x=df_train.SalePrice);
plt.subplot(2,2,2)
sns.distplot(df_train.SalePrice);
plt.subplot(2,2,3)
sns.boxplot(df_train.SalePrice);
plt.suptitle("SalePrice - distribution");

# Model Evaluation

df_model_results=pd.DataFrame(columns=["ModelName","TrainScore","TestScore"])

def store_model_results(modl_name,train_score,test_score):
    global df_model_results
    row_loc=df_model_results.shape[0]+1
    df_model_results.loc[row_loc,["ModelName","TrainScore","TestScore"]]=[modl_name,train_score,test_score]
    
    #Function for providing generalized results for regression model
def evaluate_model(model,x_train,y_train,x_test,y_test):
    model.fit(x_train,y_train)
    model_name=model.__class__.__name__
    train_score=model.score(x_train,y_train)
    test_score=model.score(x_test,y_test)
    print (f"Training score: {train_score} \nTesting score: {test_score}")
    
    store_model_results(model_name,train_score,test_score )
    y_pred=model.predict(x_test)
    print("Prediction completed")
    df=pd.DataFrame({"Actual": y_test,
                     "Predicted":y_pred})
    
    #Applying square function to transform to original target variable.
    df=df.apply(np.square)
    #Finding the difference between original and predicted
    df["difference"]=df.Predicted-df.Actual
    df.reset_index(inplace=True)
    #Plot actual vs predicted
    plt.figure(figsize=(20,10));
    sns.scatterplot(data=df,x="index",y="Actual",color='lightgrey',label=["Actual"]);
    sns.lineplot(data=df,x="index",y="Predicted",color='red',label=["Predicted"]);
    plt.legend(loc="right",bbox_to_anchor=(1.1,1));
    plt.title(model_name+" -Actual vs Predicted");
    plt.show()
    return model

df_model_results=df_model_results.iloc[0:0]
for model in  [LinearRegression(),
               Lasso(),Ridge(),
               ElasticNet(),
               XGBRegressor(learning_rate=0.1,
                            max_depth=3,
                            colsample_bytree=0.2,
                            subsample=0.7,
                            n_estimators=300)]:
    store_model_results(model.__class__.__name__,
                       'CrossValidation', 
                       cross_val_score(model,x,y,cv=3).mean(),
                       )
df_model_results.sort_values("TestScore",ascending=False,inplace=True)
df_model_results


#Model: LinearRegression
lr_model=evaluate_model(LinearRegression(normalize=True),x_train,y_train,x_test,y_test)

#Model: Lasso Regression
lasso_reg=evaluate_model(Lasso(alpha=.01,normalize=True),x_train,y_train,x_test,y_test)

#Model: Ridge Regression
ridge_reg=evaluate_model(Ridge(normalize=True),x_train,y_train,x_test,y_test)

#Model: ElasticNet Regression
el=evaluate_model(ElasticNet(normalize=False),x_train,y_train,x_test,y_test)


#Model: XtremeGradientBoosting
xgb_reg=evaluate_model(XGBRegressor(learning_rate=0.1,
                            max_depth=3,
                            colsample_bytree=0.2,
                            subsample=0.7,
                            n_estimators=300),x_train,y_train,x_test,y_test)



 
 


    