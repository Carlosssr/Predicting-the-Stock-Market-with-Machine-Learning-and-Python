#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install yfinance')


# In[5]:


import yfinance as yf
import pandas as pd


# In[6]:


spy_ticker = yf.Ticker("SPY")


# In[7]:


spy_data = spy_ticker.history(period="max")


# In[8]:


spy_data


# In[9]:


spy_data.index


# In[10]:


spy_data.plot.line(y="Close", use_index=True)


# In[11]:


del spy_data["Dividends"]
del spy_data["Stock Splits"]


# In[12]:


spy_data["Tomorrow"] = spy_data["Close"].shift(-1)


# In[13]:


spy_data


# In[14]:


spy_data["Target"] = (spy_data["Tomorrow"] > spy_data["Close"]).astype(int)


# In[17]:


spy_data


# In[22]:


spy_data = spy_data.loc["1993-01-29":].copy()


# In[23]:


spy_data


# In[26]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = spy_data.iloc[:-100]
test = spy_data.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]

model.fit(train[predictors], train ["Target"])


# In[28]:


from sklearn.metrics import precision_score #If we guessed up did it actually go up.

preds = model.predict(test[predictors])


# In[29]:


import pandas as pd

preds = pd.Series(preds, index=test.index)


# In[30]:


precision_score(test["Target"], preds)


# In[31]:


combined = pd.concat([test["Target"], preds], axis=1)


# In[33]:


combined.plot()


# In[34]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return combined


# In[ ]:


def backtest(data, model, predictors, start=2500, step=250):
    

