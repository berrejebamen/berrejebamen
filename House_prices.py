#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib as plt


# In[2]:


df=pd.read_csv('train.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[7]:


df.drop(['Alley'],axis=1,inplace=True)


# In[8]:


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[9]:


df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[10]:


df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])


# In[11]:


df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[12]:


df.drop(['Id'],inplace=True,axis=1)


# In[13]:


df.shape


# In[14]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[15]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[16]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='gray')


# In[17]:


df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])


# In[18]:


df.dropna(inplace=True)


# In[19]:


df.shape


# In[20]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='gray')


# In[21]:


g=df.columns.to_series().groupby(df.dtypes).groups


# In[22]:


g


# In[23]:


columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


# In[24]:


len(columns)


# In[25]:


main_df=df.copy()


# In[26]:


test_df=pd.read_csv('test.csv')


# In[27]:


test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
test_df.drop(['Alley'],axis=1,inplace=True)
test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])
test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])
test_df.drop(['GarageYrBlt'],axis=1,inplace=True)
test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
test_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])
test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])


# In[28]:


test_df.shape


# In[29]:


test_df.drop(['Id'],inplace=True,axis=1)


# In[30]:


test_df.shape


# In[31]:


test_df.to_csv('formulatedtest.csv')


# In[32]:


test_df.shape


# In[33]:


final_df=pd.concat([df,test_df],axis=0)


# In[34]:


final_df['SalePrice']


# In[35]:


final_df.shape


# In[36]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
    df_final=pd.concat([final_df,df_final],axis=1)
    return df_final


# In[37]:


test_df.shape


# In[38]:


final_df=category_onehot_multcols(columns)


# In[39]:


final_df=final_df.loc[:,~final_df.columns.duplicated()]


# In[40]:


final_df.head()


# In[41]:


final_df.shape


# In[42]:


df_train=final_df.iloc[:1422,:]
df_test=final_df.iloc[1422:,:]


# In[43]:


df_train.shape


# In[44]:


df_test.drop(['SalePrice'],axis=1,inplace=True)


# In[45]:


df_test.shape


# In[46]:


X_train=df_train.drop(['SalePrice'],axis=1)
y_train=df_train['SalePrice']


# In[47]:



import xgboost
classifier=xgboost.XGBRegressor()
classifier.fit(X_train,y_train)


# In[48]:


import pickle
filename='finalized_model.pkl'
pickle.dump(classifier,open(filename,'wb'))


# In[49]:


y_pred=classifier.predict(df_test)


# In[50]:


y_pred


# In[53]:


pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('sample_submission.csv')
dataset=pd.concat([sub_df['Id'],pred],axis=1)
dataset.columns=['Id','SalePrice']
dataset.to_csv('sample_submission.csv',index=True)

