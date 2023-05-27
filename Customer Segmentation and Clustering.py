#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("C:/Users/ADERONKE/Downloads/Mall_Customers.csv")


# # UNIVARIATE ANALYSIS

# In[3]:


df.describe()


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


sns.distplot(df['Annual Income (k$)'])


# In[7]:


sns.displot(df['Annual Income (k$)'])


# In[8]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[9]:


sns.kdeplot(df['Annual Income (k$)'],shade=True);


# In[10]:


sns.kdeplot(df['Annual Income (k$)'],shade=True,hue=df['Gender']);


# In[11]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
sns.kdeplot(df['Annual Income (k$)'],shade=True,hue=df['Gender']);    


# In[12]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i])


# In[13]:


df['Gender'].value_counts()


# In[14]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[15]:


sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)')


# In[16]:


sns.pairplot(df)


# In[17]:


df=df.drop('CustomerID',axis=1)
sns.pairplot(df)


# In[18]:


#df=df.drop('CustomerID',axis=1)
sns.pairplot(df,hue='Gender')


# In[19]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()


# In[20]:


df.corr()


# In[21]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# # Clustering-Univariate,Bivariate,Multivariate

# In[22]:


clustering1 = KMeans()


# In[23]:


clustering1 = KMeans(n_clusters=3)


# In[24]:


clustering1.fit(df[['Annual Income (k$)']])


# In[25]:


clustering1.labels_


# In[26]:


df['Income cluster'] = clustering1.labels_
df.head()


# In[27]:


df['Income cluster'].value_counts()


# In[29]:


inertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[30]:


inertia_scores


# In[31]:


plt.plot(range(1,11),inertia_scores)


# # Bivariate Clustering

# In[32]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[33]:


inertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
    plt.plot(range(1,11),inertia_scores)


# In[34]:


sns.scatterplot(data=df, x = 'Annual Income (k$)',y='Spending Score (1-100)')


# In[35]:


clustering2.cluster_centers_


# In[43]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers


# In[44]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[81]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x = 'Annual Income (k$)', y='Spending Score (1-100)',hue='Spending and Income Cluster',palette ='tab10')
plt.savefig('Clustering_bivaraiate.png')


# In[50]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[52]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()


# In[55]:


#Multivariate clustering
from sklearn.preprocessing import StandardScaler


# In[56]:


df.head()


# In[58]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[59]:


dff.columns


# In[60]:


dff = dff[['Age', 'Annual Income (k$)','Spending Score (1-100)','Gender_Male']]
dff.head()


# In[72]:


dff 


# In[78]:


df


# In[80]:


df.to_csv('Clustering.csv')

