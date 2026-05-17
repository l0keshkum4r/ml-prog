import cors from "cors";
import express from "express";

const app = express();

app.use(express.json());
app.use(cors());

app.use("/p1", (req, res) => {
  res.send(`# PROGRAM 1
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r'/content/housing.csv')
df['total_bedrooms'].fillna(df['total_bedrooms'].median(),inplace=True)
df[df.columns[2:7]]=df[df.columns[2:7]].astype(int)

for c in df.select_dtypes(include=np.number):
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    df[c].hist(bins=60,edgecolor='black')
    plt.title(c)
    plt.subplot(122)
    sns.boxplot(x=df[c],color='blue')
    plt.title(c)
    plt.show()`);
});

app.use("/p2", (req, res) => {
  res.send(`# PROGRAM 2

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import fetch_california_housing

d=fetch_california_housing()
df=pd.DataFrame(d.data,columns=d.feature_names)
df['Target']=d.target
df.hist(figsize=(12,8),bins=30,edgecolor='black')
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.show()
sns.pairplot(df[['MedInc','HouseAge','AveRooms','Target']],diag_kind='kde')
plt.show()`);
});

app.use("/p3", (req, res) => {
  res.send(`# PROGRAM 3

import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris=datasets.load_iris()
X,y=iris.data,iris.target
X=StandardScaler().fit_transform(X)
cov=np.cov(X.T)
e_val,e_vec=np.linalg.eig(cov)

print(cov)
print(e_val)
print(e_vec)

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')

for i,c in enumerate(['red','green','blue']):
    ax.scatter(X[y==i,0],X[y==i,1],X[y==i,2],color=c)

plt.show()
U,S,Vt=np.linalg.svd(X,full_matrices=False)
print(S)
pca=PCA(n_components=2)
Xp=pca.fit_transform(X)

print(pca.explained_variance_ratio_)
plt.figure(figsize=(8,6))

for i,c in enumerate(['red','green','blue']):
    plt.scatter(Xp[y==i,0],Xp[y==i,1],color=c)

plt.grid()
plt.show()

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')

for i,c in enumerate(['red','green','blue']):
    ax.scatter(X[y==i,0],X[y==i,1],X[y==i,2],color=c)

for i in range(3):
    ax.quiver(0,0,0,e_vec[i,0],e_vec[i,1],e_vec[i,2],color='black')

plt.show()`);
});

app.use("/p4", (req, res) => {
  res.send(`# PROGRAM 4

import pandas as pd

df=pd.read_csv(r'/content/find.csv')
x,y=df.iloc[:,:-1].values,df.iloc[:,-1].values

for i,v in enumerate(y):
    if v=='Yes':
        h=x[i].copy()
        break

for i,v in enumerate(y):
    if v=='Yes':
        for j in range(len(h)):
            if h[j]!=x[i][j]:
                h[j]='?'

print("Most Specific Hypothesis:",h)`);
});

app.use("/p5", (req, res) => {
  res.send(`# PROGRAM 5

import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
v=np.random.rand(100)

l=['Class1' if i<=0.5 else 'Class2' for i in v[:50]]+[None]*50

df=pd.DataFrame({
    'Point':[f'x{i+1}' for i in range(100)],
    'Value':v,
    'Label':l
})

df[['Value']].hist(figsize=(10,6),bins=30,edgecolor='black')
plt.show()

train=df[df.Label.notna()]
test=df[df.Label.isna()]

X,y=train[['Value']],train['Label']
Xt=test[['Value']]

true=['Class1' if i<=0.5 else 'Class2' for i in v[50:]]

for k in [1,2,3,4,5,20,30]:
    m=KNeighborsClassifier(n_neighbors=k)
    m.fit(X,y)
    p=m.predict(Xt)
    print(f'Accuracy for k={k}: {accuracy_score(true,p)*100:.2f}%')`);
});

app.use("/p6", (req, res) => {
  res.send(`# PROGRAM 6

import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

g=lambda x,xq,t:np.exp(-((x-xq)**2)/(2*t**2))

def lwr(X,y,xq,t):
    Xb=np.c_[np.ones(len(X)),X]
    W=np.diag(g(X,xq,t))
    th=np.linalg.pinv(Xb.T@W@Xb)@Xb.T@W@y
    return np.array([1,xq])@th

X=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,3,2,4,3.5,5,6,7,6.5,8])
Xq=np.linspace(1,10,100)
lr=LinearRegression().fit(X.reshape(-1,1),y)
plt.figure(figsize=(12,8))
plt.scatter(X,y,color='blue')
plt.plot(
    Xq,
    lr.predict(Xq.reshape(-1,1)),
    'k--',
    label='Linear Regression'
)

for t,c in zip([0.1,0.5,1,5,10],['red','green','purple','orange','brown']):
    plt.plot(
        Xq,
        [lwr(X,y,i,t) for i in Xq],
        color=c,
        label=f'LWR (τ={t})'
    )

plt.legend()
plt.show()`);
});

app.use("/p7a", (req, res) => {
  res.send(`# PROGRAM 7A

import pandas as pd,numpy as np,matplotlib.pyplot as plt,seaborn as sns,warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv(r'/content/Boston housing dataset.csv')

df['CRIM'].fillna(df['CRIM'].mean(),inplace=True)
df['ZN'].fillna(df['ZN'].mean(),inplace=True)
df['CHAS'].fillna(df['CHAS'].mode()[0],inplace=True)
df['INDUS'].fillna(df['INDUS'].mean(),inplace=True)
df['AGE'].fillna(df['AGE'].median(),inplace=True)
df['LSTAT'].fillna(df['LSTAT'].median(),inplace=True)

df['CHAS']=df['CHAS'].astype(int)

for c in df.columns:
    plt.figure(figsize=(8,3))
    plt.subplot(121)
    df[c].hist(edgecolor='black')
    plt.subplot(122)
    plt.boxplot(df[c],vert=False)
    plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.show()

X=StandardScaler().fit_transform(df.drop('MEDV',axis=1))
y=df['MEDV']

xtr,xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)

m=LinearRegression().fit(xtr,ytr)
p=m.predict(xte)

mse=mean_squared_error(yte,p)
print(mse)
print(np.sqrt(mse))
print(r2_score(yte,p))`);
});

app.use("/p7b", (req, res) => {
  res.send(`# PROGRAM 7B

import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns,warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df=sns.load_dataset('mpg')
df['horsepower'].fillna(df['horsepower'].median(),inplace=True)

num=df.select_dtypes(include=['int','float']).columns
cat=df.select_dtypes(include=['object']).columns

for c in num:
    plt.figure(figsize=(8,3))
    plt.subplot(121)
    df[c].hist(edgecolor='black')
    plt.subplot(122)
    plt.boxplot(df[c],vert=False)
    plt.show()

for c in cat:
    plt.figure(figsize=(6,6))
    sns.countplot(x=c,data=df)
    plt.xticks(rotation=90)
    plt.show()

sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm',fmt='.2f')
plt.show()

X=df[['horsepower']]
y=df['mpg']

xtr,xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
poly=PolynomialFeatures(2)
m=LinearRegression().fit(poly.fit_transform(xtr),ytr)
p=m.predict(poly.transform(xte))
xr=np.linspace(X.min(),X.max(),100).reshape(-1,1)
plt.scatter(X,y,color='blue')
plt.plot(xr,m.predict(poly.transform(xr)),color='red')
plt.show()

mse=mean_squared_error(yte,p)
print(mse)
print(np.sqrt(mse))
print(r2_score(yte,p))`);
});

app.use("/p8", (req, res) => {
  res.send(`# PROGRAM 8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from IPython.display import Image
import pydotplus
import math

data = pd.read_csv(r'/content/Breast Cancer Dataset.csv')
pd.set_option('display.max_columns', None)
df = data.drop(['id'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

def entropy(col):
    counts = col.value_counts()
    prob = counts / len(col)
    return -sum(prob * prob.apply(math.log2))

def conditional_entropy(data, feature, target):
    values = data[feature].unique()
    w_entropy = 0
    for v in values:
        subset = data[data[feature] == v]
        w_entropy += (
            len(subset) / len(data)
        ) * entropy(subset[target])

    return w_entropy

def information_gain(data, feature, target):
    return entropy(data[target]) - conditional_entropy(
        data,
        feature,
        target
    )
print("\nInformation Gain Values\n")

for f in X.columns:
    print( f"{f} :", information_gain(df, f, 'diagnosis'))

dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=X.columns,
    filled=True,
    rounded=True
)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

plt.figure(figsize=(18,10))
plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    class_names=['Benign', 'Malignant']
)
plt.show()

y_pred = model.predict(X_test)
print("\nAccuracy :", accuracy_score(y_test, y_pred) * 100)
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))


new = [[
12.5,19.2,80.0,500.0,0.085,0.1,0.05,0.02,0.17,0.06,
0.4,1.0,2.5,40.0,0.006,0.02,0.03,0.01,0.02,0.003,
16.0,25.0,105.0,900.0,0.13,0.25,0.28,0.12,0.29,0.08
]]

pred = model.predict(new)
print("\nPrediction :", "Benign" if pred[0] == 0 else "Malignant")`);
});

app.use("/p9", (req, res) => {
  res.send(`# PROGRAM 9
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

data = fetch_olivetti_faces()

print("Data Shape :", data.data.shape)
print("Target Shape :", data.target.shape)
print("Unique Persons :", len(np.unique(data.target)))
print("Image Size :", data.images.shape[1], "x", data.images.shape[2])


fig, ax = plt.subplots(5, 5, figsize=(10,10))

for i, a in enumerate(ax.ravel()):
    a.imshow(data.images[i], cmap='gray')
    a.set_title(f"Person {data.target[i]}")
    a.axis('off')

plt.suptitle("Sample Faces")
plt.show()

X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

print("x_train :", x_train.shape)
print("x_test  :", x_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

print("\nGaussian Naive Bayes Results")
print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)

print("\nMultinomial Naive Bayes Results")
print("--------------------------------")
print("Accuracy :", round(accuracy_score(y_test, y_pred2) * 100, 2), "%")
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred2))
print("\nClassification Report")
print(classification_report(y_test, y_pred2))

mis = np.where(y_pred2 != y_test)[0]
print("\nNumber of Misclassified Images :", len(mis))
plt.figure(figsize=(15,5))

for i in range(min(5, len(mis))):
    idx = mis[i]
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[idx].reshape(64,64), cmap='gray')
    plt.title(f"T:{y_test[idx]}\nP:{y_pred2[idx]}")
    plt.axis('off')

plt.suptitle("Misclassified Images")
plt.show()

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
prob = mnb.predict_proba(x_test)

print("\nAUC Scores")
print("----------------")

for i in range(y_test_bin.shape[1]):
    auc = roc_auc_score(y_test_bin[:, i], prob[:, i])
    print(f"Class {i} AUC : {auc:.2f}")

plt.figure(figsize=(12,8))

for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(x_test[i].reshape(64,64), cmap='gray')
    plt.title(f"T:{y_test[i]}  P:{y_pred2[i]}")
    plt.axis('off')

plt.suptitle("Face Prediction Results")
plt.show()`);
});

app.use("/p10", (req, res) => {
  res.send(`# PROGRAM 10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

y_kmeans = kmeans.fit_predict(X_scaled)

print("Confusion Matrix\n")
print(confusion_matrix(y, y_kmeans))
print("\nClassification Report\n")
print(classification_report(y, y_kmeans))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df = pd.DataFrame(
    X_pca,
    columns=['PC1', 'PC2']
)

df['Cluster'] = y_kmeans
df['True Label'] = y

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='PC1',
    y='PC2',
    hue='Cluster',
    palette='Set1',
    s=100,
    edgecolor='black',
    alpha=0.7
)
plt.title("KMeans Clustering")
plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x='PC1',
    y='PC2',
    hue='True Label',
    palette='coolwarm',
    s=100,
    edgecolor='black',
    alpha=0.7
)
plt.title("True Labels")
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='PC1',
    y='PC2',
    hue='Cluster',
    palette='Set1',
    s=100,
    edgecolor='black',
    alpha=0.7
)

centers = pca.transform(
    kmeans.cluster_centers_
)
plt.scatter(
    centers[:,0],
    centers[:,1],
    s=300,
    c='red',
    marker='X',
    label='Centroids'
)
plt.title("Clusters with Centroids")
plt.legend()
plt.show()`);
});

// Export app for testing or further usage
export default app;
