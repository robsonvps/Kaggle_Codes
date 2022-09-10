#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[23]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# In[3]:


train.head()


# In[4]:


def transformar_sexo(valor):
        if valor == 'female':
            return 1
        else:
            return 0
        
train['Sex_binario']=train['Sex'].map(transformar_sexo)


# In[5]:


variaveis=['Sex_binario','Age','Pclass', 'SibSp', 'Parch','Fare']


# In[6]:


X=train[variaveis]
y=train['Survived']


# In[7]:


X=X.fillna(-1)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[9]:


X_treino, X_valid, y_treino, y_valid = train_test_split(X, y, test_size=0.5)


# In[10]:


X_treino.head()


# In[11]:


train['Sex'].value_counts()


# In[12]:


from sklearn.model_selection import RepeatedKFold


# In[14]:


resultados = []

kf=RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)

for linhas_treino, linhas_valid in kf.split(X):
    print("Treino:", linhas_treino.shape[0])
    print("Valid", linhas_valid.shape[0])

    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)
    modelo.fit(X_treino, y_treino)

    p=modelo.predict(X_valid)

    acc = np.mean(y_valid == p)
    resultados.append(acc)
    print("Acc", acc)
    print()


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


# In[16]:


resultados


# In[17]:


np.mean(resultados)


# In[18]:


pylab.hist(resultados)


# In[19]:


p = (X_valid['Sex_binario'] == 1).astype(np.int64)
np.mean(y_valid == p)


# In[20]:


modelo.fit(X,y)


# In[24]:


test['Sex_binario']=test['Sex'].map(transformar_sexo)


# In[25]:


X_prev=test[variaveis]
X_prev=X_prev.fillna(-1)
X_prev.head()


# In[26]:


p=modelo.predict(test[variaveis].fillna(-1))


# In[27]:


p


# In[28]:


#Validação Cruzada


# In[29]:


test.head()


# In[31]:


sub=pd.Series(p, index=test['PassengerId'], name='Survived')
sub.shape


# In[32]:


sub.to_csv('final_model.csv', header=True)


# In[33]:


get_ipython().system('head -n10 segundo_modelo.csv')


# In[ ]:




