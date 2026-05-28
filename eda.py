
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf

df = pd.read_csv('Customer_churn.csv')

df.head()

df.info()

df.describe()

df = df.drop('customerID', axis=1)

df.isnull().sum()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

for col in df.columns:
    print(f"Unique values for {col}:")
    print(df[col].unique())


sns.histplot(df['tenure'])

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_cols


for col in num_cols:
    plt.figure(figsize=(5, 2))
    print(sns.boxplot(x=df[col]))


# In[17]:


for col in df.columns:
    plt.figure(figsize=(5, 2))
    print(sns.countplot(x=df[col], hue=df['Churn']))


# In[18]:


for col in num_cols:
    plt.figure(figsize=(5, 3))
    sns.histplot(data=df, x=col, hue='Churn', bins=30, kde=True)
    plt.title(f"{col} distribution by Churn")
    plt.show()


# In[19]:


df['Churn'].value_counts()


# In[22]:


for col in df.columns:
    print(f"Unique values for {col}:")
    print(df[col].unique())


# In[23]:


df.head()


# In[24]:


df['gender']=df['gender'].replace({'Male':0, 'Female':1})
df[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']]=df[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']].replace({'Yes':1, 'No':0})


# In[25]:


df.head()


# In[26]:


X=df.drop('Churn', axis=1)
y=df['Churn']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45, stratify=y)


# In[27]:


X_train


# In[28]:


df.info()


# In[29]:


ohe_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract'
]


# In[30]:


ohe=OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_ohe=ohe.fit_transform(X_train[ohe_cols])
X_test_ohe=ohe.transform(X_test[ohe_cols])


# In[31]:


ohe_cols_new=ohe.get_feature_names_out(ohe_cols)

X_train_ohe = pd.DataFrame(
    X_train_ohe,
    columns=ohe_cols_new,
    index=X_train.index
)

X_test_ohe = pd.DataFrame(
    X_test_ohe,
    columns=ohe_cols_new,
    index=X_test.index
)


# In[32]:


X_train = X_train.drop(columns=ohe_cols)
X_test  = X_test.drop(columns=ohe_cols)


# In[33]:


X_train = pd.concat([X_train, X_train_ohe], axis=1)
X_test  = pd.concat([X_test, X_test_ohe], axis=1)


# In[34]:


X_train.shape, X_test.shape


# In[35]:


X_train_pm=ohe.fit_transform(X_train[['PaymentMethod']])


# In[36]:


X_test_pm=ohe.transform(X_test[['PaymentMethod']])


# In[37]:


pm_cols=ohe.get_feature_names_out(['PaymentMethod'])


# In[38]:


X_train_pm = pd.DataFrame(X_train_pm, columns=pm_cols, index=X_train.index)
X_test_pm  = pd.DataFrame(X_test_pm, columns=pm_cols, index=X_test.index)


# In[39]:


X_train = X_train.drop(columns=['PaymentMethod'])
X_test  = X_test.drop(columns=['PaymentMethod'])


# In[40]:


X_train = pd.concat([X_train, X_train_pm], axis=1)
X_test  = pd.concat([X_test, X_test_pm], axis=1)


# In[41]:


X_train.info()


# In[42]:


y_train.info()


# In[43]:


y_train = y_train.map({'Yes': 1, 'No': 0})
y_test  = y_test.map({'Yes': 1, 'No': 0})


# In[44]:


y_train.info()


# In[45]:


X_train.head()


# In[46]:


#Identified missing TotalCharges for customers with 0 tenure.
# imputed with 0 to represent new accounts before first billing

X_train['TotalCharges'] = X_train['TotalCharges'].fillna(0)
X_test['TotalCharges']  = X_test['TotalCharges'].fillna(0)


# In[47]:


from sklearn.preprocessing import StandardScaler

scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = StandardScaler()

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])


# In[48]:


X_train[scale_cols].describe()


# In[49]:


y_train.value_counts(normalize=True)


# In[50]:


X_train.isnull().sum()

from sklearn.utils import resample
train_df = pd.concat([X_train, y_train], axis=1)

#seperate them
majority = train_df[train_df['Churn'] == 0]
minority = train_df[train_df['Churn'] == 1]



# In[53]:


from sklearn.utils import resample

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)


# In[54]:


train_upsampled = pd.concat([majority, minority_upsampled])
train_upsampled = train_upsampled.sample(frac=1, random_state=42)


X_train_up = train_upsampled.drop('Churn', axis=1)
y_train_up = train_upsampled['Churn']


# In[55]:


print(X_train_up.columns)


# In[56]:


#logistic regression without smote

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_up, y_train_up)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]


# In[57]:


# tuning the threshold to find more churners.

thresholds=np.arange(0.25, 0.6, 0.05)
for t in thresholds:
    y_pred_thr= (y_prob>t).astype(int)
    print(f"threshold: {t}")
    print(classification_report(y_test, y_pred_thr))


# In[58]:


y_prob = lr.predict_proba(X_test)[:, 1]
y_pred_final = (y_prob > 0.40).astype(int)


# In[59]:


print(confusion_matrix(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# In[60]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(class_weight='balanced')
rfc.fit(X_train_up, y_train_up)

y_pred1=rfc.predict(X_test)
y_prob1 = rfc.predict_proba(X_test)[:, 1]
y_prob2= y_prob


# In[61]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob1))


# In[62]:


from sklearn.model_selection import GridSearchCV

# Define the parameters to test
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

# Initialize the grid search
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=45), 
                           param_grid=param_grid, 
                           cv=3, scoring='roc_auc', n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train_up, y_train_up)

# See the best parameters
print("Best Parameters:", grid_search.best_params_)
best_rfc = grid_search.best_estimator_


# In[63]:


import joblib

#the best parmeter found
best_rfc = RandomForestClassifier(
    n_estimators=200, 
    min_samples_split=2, 
    max_depth=None, 
    class_weight='balanced', 
    random_state=45
)

best_rfc.fit(X_train_up, y_train_up)


from sklearn.preprocessing import OneHotEncoder

# 1. Prepare the raw categorical data for fitting
# We use the original dataframe but filter for the columns we need
ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']

# 2. Re-fit General Encoder
ohe_gen = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohe_gen.fit(df[ohe_cols]) 

# 3. Re-fit Payment Encoder
ohe_pay = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohe_pay.fit(df[['PaymentMethod']])

# 4. SAVE EVERYTHING
joblib.dump(best_rfc, 'final_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(ohe_gen, 'ohe_general.pkl')
joblib.dump(ohe_pay, 'ohe_payment.pkl')

print("✅ Success! All 4 files saved correctly.")


print("Success: Best model, Scaler, and Encoder saved separately!")


from tensorflow.python import pywrap_tensorflow

from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy', 'Recall', 'Precision', 'AUC'])

model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2)

y_prob_ann = model.predict(X_test).ravel()


for t in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    y_pred = (y_prob_ann > t).astype(int)
    print("Threshold:", t)
    print(classification_report(y_test, y_pred))
