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

app.use("/prog1", (req, res) => {
  res.send(`import string
import numpy as np

ALPHABET = string.ascii_uppercase
MOD = 26

# CAESAR CIPHER

def caesar_encrypt(text, shift):
    result = ""
    for char in text.upper():
        if char in ALPHABET:
            index = (ALPHABET.index(char) + shift) % MOD
            result += ALPHABET[index]
        else:
            result += char
    return result


def caesar_decrypt(cipher, shift):
    return caesar_encrypt(cipher, -shift)


#  SUBSTITUTION CIPHER

def substitution_encrypt(text, key):
    """
    key must be 26 unique uppercase letters
    Example:
    QWERTYUIOPASDFGHJKLZXCVBNM
    """
    text = text.upper()
    result = ""
    for char in text:
        if char in ALPHABET:
            result += key[ALPHABET.index(char)]
        else:
            result += char
    return result


def substitution_decrypt(cipher, key):
    result = ""
    for char in cipher:
        if char in key:
            result += ALPHABET[key.index(char)]
        else:
            result += char
    return result


# HILL CIPHER (2x2 Matrix Only)

def mod_inverse(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None


def process_text_hill(text):
    text = ''.join([c for c in text. Upper() if c in ALPHABET])
    if len(text) % 2 != 0:
        text += 'X'
    return text


def hill_encrypt(text, key_matrix):
    text = process_text_hill(text)
    cipher = ""

    for i in range(0, len(text), 2):
        pair = text[i:i+2]
        vector = [ALPHABET.index(pair[0]), ALPHABET.index(pair[1])]
        result = np.dot(key_matrix, vector) % MOD
        cipher += ALPHABET[result[0]] + ALPHABET[result[1]]

    return cipher


def hill_decrypt(cipher, key_matrix):
    det = int(np.round(np.linalg.det(key_matrix)))
    det = det % MOD
    det_inv = mod_inverse(det, MOD)

    if det_inv is None:
        raise ValueError("Key matrix is not invertible")

    adjugate = np.round(det * np.linalg.inv(key_matrix)).astype(int) % MOD
    inverse_matrix = (det_inv * adjugate) % MOD

    plain = ""

    for i in range(0, len(cipher), 2):
        pair = cipher[i:i+2]
        vector = [ALPHABET.index(pair[0]), ALPHABET.index(pair[1])]
        result = np.dot(inverse_matrix, vector) % MOD
        plain += ALPHABET[result[0]] + ALPHABET[result[1]]

    return plain


#  VIGENERE CIPHER

def vigenere_encrypt(text, key):
    text = text.upper()
    key = key.upper()
    result = ""
    key_index = 0

    for char in text:
        if char in ALPHABET:
            shift = ALPHABET.index(key[key_index % len(key)])
            index = (ALPHABET.index(char) + shift) % MOD
            result += ALPHABET[index]
            key_index += 1
        else:
            result += char
    return result


def vigenere_decrypt(cipher, key):
    cipher = cipher.upper()
    key = key.upper()
    result = ""
    key_index = 0

    for char in cipher:
        if char in ALPHABET:
            shift = ALPHABET.index(key[key_index % len(key)])
            index = (ALPHABET.index(char) - shift) % MOD
            result += ALPHABET[index]
            key_index += 1
        else:
            result += char
    return result


# MAIN FUNCTION

def main():
    print("\n===== CLASSICAL CIPHERS =====")
    print("1. Caesar Cipher")
    print("2. Substitution Cipher")
    print("3. Hill Cipher (2x2)")
    print("4. Vigenere Cipher")

    choice = int(input("Choose algorithm (1-4): "))

    text = input("Enter text: ")

    if choice == 1:
        shift = int(input("Enter shift value: "))
        enc = caesar_encrypt(text, shift)
        dec = caesar_decrypt(enc, shift)

    elif choice == 2:
        key = input("Enter 26-letter substitution key: ").upper()
        enc = substitution_encrypt(text, key)
        dec = substitution_decrypt(enc, key)

    elif choice == 3:
        print("Enter 2x2 key matrix values:")
        a = int(input("a: "))
        b = int(input("b: "))
        c = int(input("c: "))
        d = int(input("d: "))
        key_matrix = np.array([[a, b], [c, d]])
        enc = hill_encrypt(text, key_matrix)
        dec = hill_decrypt(enc, key_matrix)

    elif choice == 4:
        key = input("Enter keyword: ")
        enc = vigenere_encrypt(text, key)
        dec = vigenere_decrypt(enc, key)

    else:
        print("Invalid choice")
        return

    print("\nEncrypted Text :", enc)
    print("Decrypted Text :", dec)


if __name__ == "__main__":
    main()
`);
});

app.use("/prog2", (req, res) => {
  res.send(`#S-DES IMPLEMENTATION

# Permutation Tables
P10 = [3,5,2,7,4,10,1,9,8,6]
P8  = [6,3,7,4,8,5,10,9]
IP  = [2,6,3,1,4,8,5,7]
IP_INV = [4,1,3,5,7,2,8,6]
EP  = [4,1,2,3,2,3,4,1]
P4  = [2,4,3,1]

# S-Boxes
S0 = [[1,0,3,2],
      [3,2,1,0],
      [0,2,1,3],
      [3,1,3,2]]

S1 = [[0,1,2,3],
      [2,0,1,3],
      [3,0,1,0],
      [2,1,0,3]]

# Basic Functions 

def permute(bits, table):
    return ''.join(bits[i-1] for i in table)

def left_shift(bits, n):
    return bits[n:] + bits[:n]

def xor(a, b):
    return ''.join('0' if a[i] == b[i] else '1' for i in range(len(a)))

def sbox(bits, box):
    row = int(bits[0] + bits[3], 2)
    col = int(bits[1] + bits[2], 2)
    return format(box[row][col], '02b')

# Key Generation

def generate_keys(key):
    key = permute(key, P10)
    
    left = left_shift(key[:5], 1)
    right = left_shift(key[5:], 1)
    K1 = permute(left + right, P8)

    left = left_shift(left, 2)
    right = left_shift(right, 2)
    K2 = permute(left + right, P8)

    return K1, K2

# Round Function 

def fk(bits, key):
    left = bits[:4]
    right = bits[4:]

    temp = permute(right, EP)
    temp = xor(temp, key)

    s0_out = sbox(temp[:4], S0)
    s1_out = sbox(temp[4:], S1)

    temp = permute(s0_out + s1_out, P4)
    left = xor(left, temp)

    return left + right

#Encryption 

def encrypt(plaintext, key):
    K1, K2 = generate_keys(key)

    bits = permute(plaintext, IP)
    bits = fk(bits, K1)

    bits = bits[4:] + bits[:4]  # Swap halves

    bits = fk(bits, K2)

    ciphertext = permute(bits, IP_INV)
    return ciphertext

# Decryption 

def decrypt(ciphertext, key):
    K1, K2 = generate_keys(key)

    bits = permute(ciphertext, IP)
    bits = fk(bits, K2)

    bits = bits[4:] + bits[:4]  # Swap halves

    bits = fk(bits, K1)

    plaintext = permute(bits, IP_INV)
    return plaintext


key = input("Enter 10-bit key: ")
plaintext = input("Enter 8-bit plaintext: ")

cipher = encrypt(plaintext, key)
print("Ciphertext:", cipher)
print("Decrypted :", decrypt(cipher, key))
`);
});

app.use("/prog3", (req, res) => {
  res.send(`import random
from math import gcd

def mod_inverse(e, phi):
    for d in range(1, phi):
        if (d * e) % phi == 1:
            return d

def rsa_keygen():
    p, q = 61, 53
    n = p * q
    phi = (p-1)*(q-1)
    e = 17
    d = mod_inverse(e, phi)
    return (e, n), (d, n)

def rsa_encrypt(msg, pub):
    e, n = pub
    return pow(msg, e, n)

def rsa_decrypt(cipher, priv):
    d, n = priv
    return pow(cipher, d, n)


pub, priv = rsa_keygen()
cipher = rsa_encrypt(65, pub)
plain = rsa_decrypt(cipher, priv)
print("Encrypted:", cipher)
print("Decrypted:", plain)
`);
});

app.use("/prog4", (req, res) => {
  res.send(`def diffie_hellman():
    p = 23
    g = 5
    a = 6
    b = 15
    A = pow(g, a, p)
    B = pow(g, b, p)

    shared_A = pow(B, a, p)
    shared_B = pow(A, b, p)
    print("Shared Key:", shared_A)
diffie_hellman()
`);
});

app.use("/prog5", (req, res) => {
  res.send(`import hashlib

def sha1_digest(text):
    return hashlib.sha1(text.encode()).hexdigest()

print("SHA-1:", sha1_digest("Hello World"))
`);
});

app.use("/prog6", (req, res) => {
  res.send(`def sign(message, priv):
    digest = int(hashlib.sha1(message.encode()).hexdigest(), 16)
    return pow(digest, priv[0], priv[1])

def verify(message, signature, pub):
    digest = int(hashlib.sha1(message.encode()).hexdigest(), 16)
    return digest == pow(signature, pub[0], pub[1])

signature = sign("Hello", priv)
print("Verified:", verify("Hello", signature, pub))
`);
});

app.use("/prog7", (req, res) => {
  res.send(`Step 1: Install required package
pip install cryptography

Step 2: SSL Server
# ssl_server_auto.py

import socket
import ssl
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime

# Generate private key
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# Generate self-signed certificate
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, u"IN"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Karnataka"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, u"Bangalore"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Test Org"),
    x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
])

cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
    .sign(key, hashes.SHA256())
)

# Save certificate and key
with open("server.key", "wb") as f:
    f.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))

with open("server.crt", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

# SSL server
HOST = '127.0.0.1'
PORT = 8443

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain("server.crt", "server.key")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((HOST, PORT))
    sock.listen(5)
    print("Secure server running...")

    with context.wrap_socket(sock, server_side=True) as ssock:
        conn, addr = ssock.accept()
        print("Connected:", addr)

        data = conn.recv(1024)
        print("Client says:", data.decode())

        conn.send(b"Hello Client, secure connection successful!")

Step 3: SSL Client

# ssl_client.py

import socket
import ssl

HOST = '127.0.0.1'
PORT = 8443

context = ssl._create_unverified_context()

with socket.create_connection((HOST, PORT)) as sock:
    with context.wrap_socket(sock, server_hostname=HOST) as ssock:
        print("SSL connection established")

        ssock.send(b"Hello Server")
        print(ssock.recv(1024).decode())
`);
});

app.use("/flutter/p1", (req, res) => {
  res.send(`import 'package:flutter/material.dart';

void main() => runApp(MaterialApp(home: CardPage()));

class CardPage extends StatelessWidget {
  @override
  Widget build(BuildContext c) {
    return Scaffold(
      body: Center(
        child: Card(
          margin: EdgeInsets.all(20),
          child: Padding(
            padding: EdgeInsets.all(15),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        "ABC TECHNOLOGIES",
                        textAlign: TextAlign.center,
                        style: TextStyle(fontWeight: FontWeight.bold,fontSize: 18),
                      ),
                    ),
                    Icon(Icons.business,size: 40)
                  ],
                ),
                SizedBox(height: 10),
                Text("Snehal Rathi",
                    style: TextStyle(fontWeight: FontWeight.bold)),
                Text("Software Engineer"),
                Divider(),
                Text("Phone: 9876543210"),
                Text("Email: snehal@abc.com"),
                Text("Fax: 020-123456"),
                Text("Website: www.abctech.com"),
                Text("Address: Pune"),
              ],
            ),
          ),
        ),
      ),
    );
  }
}`);
});

app.use("/flutter/p2", (req, res) => {
  res.send(`import 'package:flutter/material.dart';

void main() => runApp(MaterialApp(home: SignUp()));

class SignUp extends StatefulWidget {
  @override
  State<SignUp> createState() => _SignUpState();
}

class _SignUpState extends State<SignUp> {
  var u = TextEditingController(), p = TextEditingController();

  bool valid(String s) {
    return s.length >= 8 &&
        RegExp(r'[A-Z]').hasMatch(s) &&
        RegExp(r'[a-z]').hasMatch(s) &&
        RegExp(r'[0-9]').hasMatch(s) &&
        RegExp(r'[^A-Za-z0-9]').hasMatch(s);
  }

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Sign Up")),
      body: Padding(
        padding: EdgeInsets.all(20),
        child: Column(children: [
          TextField(controller: u, decoration: InputDecoration(labelText: "Username")),
          TextField(controller: p, obscureText: true,
              decoration: InputDecoration(labelText: "Password")),
          ElevatedButton(
            child: Text("SIGN UP"),
            onPressed: () {
              if (valid(p.text)) {
                Navigator.push(c,
                    MaterialPageRoute(builder: (_) => Login(u.text, p.text)));
              } else {
                ScaffoldMessenger.of(c).showSnackBar(
                    SnackBar(content: Text("Invalid Password")));
              }
            },
          )
        ]),
      ),
    );
  }
}

class Login extends StatefulWidget {
  final String u, p;
  Login(this.u, this.p);

  @override
  State<Login> createState() => _LoginState();
}

class _LoginState extends State<Login> {
  var u = TextEditingController(), p = TextEditingController();
  int tries = 2;
  bool disable = false;

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Login")),
      body: Padding(
        padding: EdgeInsets.all(20),
        child: Column(children: [
          TextField(controller: u, decoration: InputDecoration(labelText: "Username")),
          TextField(controller: p, obscureText: true,
              decoration: InputDecoration(labelText: "Password")),
          ElevatedButton(
            onPressed: disable
                ? null
                : () {
                    if (u.text == widget.u && p.text == widget.p) {
                      Navigator.push(c,
                          MaterialPageRoute(builder: (_) => Success()));
                    } else {
                      tries--;
                      if (tries == 0) {
                        setState(() => disable = true);
                        ScaffoldMessenger.of(c).showSnackBar(
                            SnackBar(content: Text("Failed Login Attempts")));
                      } else {
                        ScaffoldMessenger.of(c).showSnackBar(
                            SnackBar(content: Text("Login Failed")));
                      }
                    }
                  },
            child: Text("SIGN IN"),
          )
        ]),
      ),
    );
  }
}

class Success extends StatelessWidget {
  @override
  Widget build(BuildContext c) {
    return Scaffold(
      body: Center(
        child: Text("Successful Login",
            style: TextStyle(fontSize: 25,fontWeight: FontWeight.bold)),
      ),
    );
  }
}`);
});

app.use("/flutter/p3", (req, res) => {
  res.send(`import 'dart:async';
import 'package:flutter/material.dart';

void main() => runApp(MaterialApp(home: CounterPage()));

class CounterPage extends StatefulWidget {
  @override
  State<CounterPage> createState() => _CounterPageState();
}

class _CounterPageState extends State<CounterPage> {
  int n = 1;
  Timer? t;

  void start() {
    t = Timer.periodic(Duration(seconds: 1), (_) {
      setState(() => n++);
    });
  }

  void stop() => t?.cancel();

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Counter")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text("$n", style: TextStyle(fontSize: 40)),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(onPressed: start, child: Text("START")),
                SizedBox(width: 20),
                ElevatedButton(onPressed: stop, child: Text("STOP")),
              ],
            )
          ],
        ),
      ),
    );
  }
}`);
});

app.use("/flutter/p4", (req, res) => {
  res.send(`import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';

void main() => runApp(MaterialApp(home: TTS()));

class TTS extends StatelessWidget {
  final t = TextEditingController();
  final f = FlutterTts();

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Text To Speech")),
      body: Padding(
        padding: EdgeInsets.all(20),
        child: Column(children: [
          TextField(controller: t,
              decoration: InputDecoration(labelText: "Enter Text")),
          ElevatedButton(
            child: Text("Convert Text to Speech"),
            onPressed: () => f.speak(t.text),
          )
        ]),
      ),
    );
  }
}`);
});

app.use("/flutter/p5", (req, res) => {
  res.send(`<uses-permission android:name="android.permission.READ_CONTACTS"/>
<uses-permission android:name="android.permission.WRITE_CONTACTS"/>
<uses-permission android:name="android.permission.CALL_PHONE"/>

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:flutter_contacts/flutter_contacts.dart';
void main() {
runApp(MyApp());
}
class MyApp extends StatelessWidget {
@override
Widget build(BuildContext context) {
return MaterialApp(
debugShowCheckedModeBanner: false,
home: DialerPage(),
);
}
}
class DialerPage extends StatefulWidget {
@override
_DialerPageState createState() => _DialerPageState();
}
class _DialerPageState extends State<DialerPage> {
TextEditingController phoneController = TextEditingController();

void callNumber() async {
String number = phoneController.text;
if (number.isNotEmpty) {
final Uri callUri = Uri(scheme: 'tel', path: number);
await launchUrl(callUri);
}
}

void saveContact() async {
if (await FlutterContacts.requestPermission()) {
final contact = Contact()
..name.first = "Saved Number"
..phones = [Phone(phoneController.text)];
await contact.insert();
ScaffoldMessenger.of(context).showSnackBar(
SnackBar(content: Text("Contact Saved Successfully")),
);
}
}
@override
Widget build(BuildContext context) {
return Scaffold(
appBar: AppBar(
title: Text("Phone Dialer"),
),
body: Padding(
padding: EdgeInsets.all(20),
child: Column(
mainAxisAlignment: MainAxisAlignment.center,
children: [
TextField(
controller: phoneController,
keyboardType: TextInputType.phone,
decoration: InputDecoration(
labelText: "Enter Phone Number",
border: OutlineInputBorder(),
),
),
SizedBox(height: 30),
Row(
mainAxisAlignment: MainAxisAlignment.spaceEvenly,
children: [
ElevatedButton(
onPressed: callNumber,
child: Text("CALL"),
),
ElevatedButton(
onPressed: saveContact,
child: Text("SAVE"),
),
],
),
],
),
),
);
}
}`);
});

app.use("/flutter/p6", (req, res) => {
  res.send(`import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';

void main() => runApp(MaterialApp(home: Player()));

class Player extends StatefulWidget {
  @override
  State<Player> createState() => _PlayerState();
}

class _PlayerState extends State<Player> {
  AudioPlayer a = AudioPlayer();
  Duration d = Duration.zero, p = Duration.zero;
  bool play = false;

  @override
  void initState() {
    super.initState();
    a.onDurationChanged.listen((x) => setState(() => d = x));
    a.onPositionChanged.listen((x) => setState(() => p = x));
  }

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Media Player")),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Slider(
            value: p.inSeconds.toDouble(),
            max: d.inSeconds.toDouble(),
            onChanged: (v) => a.seek(Duration(seconds: v.toInt())),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
                  icon: Icon(Icons.fast_rewind),
                  onPressed: () => a.seek(p - Duration(seconds: 10))),
              IconButton(
                icon: Icon(play ? Icons.pause : Icons.play_arrow),
                onPressed: () async {
                  if (play)
                    await a.pause();
                  else
                    await a.play(AssetSource("audio.mp3"));
                  setState(() => play = !play);
                },
              ),
              IconButton(
                  icon: Icon(Icons.fast_forward),
                  onPressed: () => a.seek(p + Duration(seconds: 10))),
            ],
          )
        ],
      ),
    );
  }
}`);
});

app.use("/flutter/p7", (req, res) => {
  res.send(`import 'package:flutter/material.dart';

void main() => runApp(MaterialApp(home: Largest()));

class Largest extends StatefulWidget {
  @override
  State<Largest> createState() => _LargestState();
}

class _LargestState extends State<Largest> {
  var a = TextEditingController(),
      b = TextEditingController(),
      c = TextEditingController();

  String r = "";

  void find() {
    int x = int.parse(a.text),
        y = int.parse(b.text),
        z = int.parse(c.text);

    setState(() => r = "Largest: ${[x, y, z].reduce((i, j) => (i > j ? i : j))}");
  }

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Largest")),
      body: Padding(
        padding: EdgeInsets.all(20),
        child: Column(children: [
          TextField(controller: a, decoration: InputDecoration(labelText: "No 1")),
          TextField(controller: b, decoration: InputDecoration(labelText: "No 2")),
          TextField(controller: c, decoration: InputDecoration(labelText: "No 3")),
          ElevatedButton(onPressed: find, child: Text("Find")),
          Text(r, style: TextStyle(fontSize: 20))
        ]),
      ),
    );
  }
}`);
});

app.use("/flutter/p8", (req, res) => {
  res.send(`import 'package:flutter/material.dart';

void main() => runApp(MaterialApp(home: Counter()));

class Counter extends StatefulWidget {
  @override
  State<Counter> createState() => _CounterState();
}

class _CounterState extends State<Counter> {
  int n = 0;

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Counter")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text("$n", style: TextStyle(fontSize: 40)),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                    onPressed: () => setState(() => n++),
                    child: Text("Increase")),
                SizedBox(width: 20),
                ElevatedButton(
                    onPressed: () => setState(() => n--),
                    child: Text("Decrease")),
              ],
            )
          ],
        ),
      ),
    );
  }
}`);
});

app.use("/flutter/p9", (req, res) => {
  res.send(`import 'package:flutter/material.dart';

void main() => runApp(MaterialApp(home: Register()));

class Register extends StatefulWidget {
  @override
  State<Register> createState() => _RegisterState();
}

class _RegisterState extends State<Register> {
  String g = "Male";
  bool terms = false, sub = false;

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Registration")),
      body: ListView(
        padding: EdgeInsets.all(20),
        children: [
          TextField(decoration: InputDecoration(labelText: "Name")),
          TextField(decoration: InputDecoration(labelText: "Email")),
          TextField(
            obscureText: true,
            decoration: InputDecoration(labelText: "Password"),
          ),
          RadioListTile(
            title: Text("Male"),
            value: "Male",
            groupValue: g,
            onChanged: (v) => setState(() => g = v.toString()),
          ),
          RadioListTile(
            title: Text("Female"),
            value: "Female",
            groupValue: g,
            onChanged: (v) => setState(() => g = v.toString()),
          ),
          CheckboxListTile(
            title: Text("Agree Terms"),
            value: terms,
            onChanged: (v) => setState(() => terms = v!),
          ),
          CheckboxListTile(
            title: Text("Subscribe"),
            value: sub,
            onChanged: (v) => setState(() => sub = v!),
          ),
          ElevatedButton(
            child: Text("Submit"),
            onPressed: () {
              ScaffoldMessenger.of(c).showSnackBar(
                SnackBar(
                    content: Text(
                        terms ? "Registration Successful" : "Agree Terms")),
              );
            },
          )
        ],
      ),
    );
  }
}`);
});

app.use("/flutter/p10", (req, res) => {
  res.send(`import 'package:flutter/material.dart';

void main() => runApp(MaterialApp(home: Calc()));

class Calc extends StatefulWidget {
  @override
  State<Calc> createState() => _CalcState();
}

class _CalcState extends State<Calc> {
  var a = TextEditingController(), b = TextEditingController();
  String r = "";

  void calc(String op) {
    double x = double.parse(a.text), y = double.parse(b.text);

    setState(() {
      if (op == "+") r = "${x + y}";
      if (op == "-") r = "${x - y}";
      if (op == "*") r = "${x * y}";
      if (op == "/") r = y != 0 ? "${x / y}" : "Cannot divide by zero";
    });
  }

  @override
  Widget build(BuildContext c) {
    return Scaffold(
      appBar: AppBar(title: Text("Calculator")),
      body: Padding(
        padding: EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(controller: a,
                decoration: InputDecoration(labelText: "First Number")),
            TextField(controller: b,
                decoration: InputDecoration(labelText: "Second Number")),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(onPressed: () => calc("+"), child: Text("+")),
                ElevatedButton(onPressed: () => calc("-"), child: Text("-")),
                ElevatedButton(onPressed: () => calc("*"), child: Text("×")),
                ElevatedButton(onPressed: () => calc("/"), child: Text("÷")),
              ],
            ),
            SizedBox(height: 20),
            Text(r, style: TextStyle(fontSize: 25))
          ],
        ),
      ),
    );
  }
}`);
});

app.use("/nosql/p3", (req, res) => {
  res.send(`// Step 1. Create a database
use weatherDB      
// Step 2. Create collection 
	db.createCollection(“weather”)
// Step 3. Insert data into Collection
          db.weather.insertMany([
             { city: "Bangalore", date: "2026-04-20", temperature: 30 },
             { city: "Bangalore", date: "2026-04-21", temperature: 32 },
             { city: "Bangalore", date: "2026-04-22", temperature: 29 },
            { city: "Mysore", date: "2026-04-21", temperature: 35 },
             { city: "Mysore", date: "2026-04-22", temperature: 33 }
             ]);
// step 4: Find Maximum Temperature 
               db.weather.aggregate([
                       {
                            $group: {
                                  _id: null,
                                      maxTemperature: { $max: "$temperature" }
                                             }
                           }
                  ]);

// Step 5. Find Maximum Temperature City-wise
	db.weather.aggregate([
                   {
                          $group: {
                        _id: "$city",
                          maxTemperature: { $max: "$temperature" }
                                       }
                         }
               ]);
// Step 6: Find Full Document of Maximum Temperature
	db.weather.find().sort({ temperature: -1 }).limit(1);
`);
});

app.use("/nosql/p4", (req, res) => {
  res.send(`// Step 1. Create a database
use school        
// Step 2. Create collection 
	db.createCollection(“students”)
// Step 3. Insert data into Collection
	db.students.insertMany([

	  {

  	  rollNo: 1dt22is098

  	  name: "Anita",

    	age: 21,

    	department: "ECE",

   	 marks: 89

	  },

  	{

  	  rollNo: 1dt22is093,

 	   name: "Vikram",
	
 	   age: 22,

  	  department: "CSE",

    	   marks:56
	
 	 },

	  {

  	  rollNo: 1dt22is0984,

    	  name: "Sneha",

   	 age: 20,
	
    	 department: "ME",

   	 marks: 88

  	}

	])

Basic Queries
1.	db.students.find() ------- display all record
2.	db.students.find().pretty()
3.	db.students.find({ department: "CSE" })e
4.	db.students.find({ marks: { $gt: 80 } }) " 
5.	db.students.find(: 
{},
{ name: 1, marks: 1, _id: 0 }
)"CSE"
6  6. db.students.findOne({ rollNo: 1dt22is093 })
    7.      db.students.updateOne(
              { rollNo: 1dt22is093 },
              { $set: { marks: 82 } })
     8.    db.students.deleteOne({ rollNo: 1dt22is093 })
`);
});

app.use("/nosql/p5", (req, res) => {
  res.send(`// Create Database
use collegeDB

// Insert Records
db.students.insertMany([
{ roll: 1, name: "Anu", dept: "CSE", marks: 85, city: "Mumbai" },
{ roll: 2, name: "Ravi", dept: "ECE", marks: 78, city: "Delhi" },
{ roll: 3, name: "Meena", dept: "CSE", marks: 92, city: "Mumbai" },
{ roll: 4, name: "Kiran", dept: "EEE", marks: 74, city: "Chennai" },
{ roll: 5, name: "Arjun", dept: "ECE", marks: 88, city: "Delhi" }
])

// Count students by department
db.students.aggregate([
{ $group: { _id: "$dept", totalStudents: { $sum: 1 } } }
])

// Find average marks by department
db.students.aggregate([
{ $group: { _id: "$dept", avgMarks: { $avg: "$marks" } } }
])

// Sort departments by average marks
db.students.aggregate([
{ $group: { _id: "$dept", avgMarks: { $avg: "$marks" } } },
{ $sort: { avgMarks: -1 } }
])

// Find maximum marks in each department
db.students.aggregate([
{ $group: { _id: "$dept", maxMarks: { $max: "$marks" } } }
])

// Find minimum marks in each department
db.students.aggregate([
{ $group: { _id: "$dept", minMarks: { $min: "$marks" } } }
])

// Find total marks by department
db.students.aggregate([
{ $group: { _id: "$dept", totalMarks: { $sum: "$marks" } } }
])

// Find departments having average marks greater than 80
db.students.aggregate([
{ $group: { _id: "$dept", avgMarks: { $avg: "$marks" } } },
{ $match: { avgMarks: { $gt: 80 } } }
])

// Count students city-wise
db.students.aggregate([
{ $group: { _id: "$city", totalStudents: { $sum: 1 } } }
])
`);
});

app.use("/nosql/p6", (req, res) => {
  res.send(`// Step 1. Create a database
use College       
// Step 2. Create collection 
	db.createCollection(“students”)
// Step 3. Insert data into Collection
db.students.insertMany ([
{ name: "Ravi", age: 20, dept: "CSE", marks: 80 },
{name: "Anu", age: 21, dept: "ISE", marks: 75 },
{name: "Kiran", age: 22, dept: "CSE", marks: 85 },
{name: "Meena", age: 20, dept: "ECE", marks: 70}
	])
// Step 3. Display Data
          db.students.find()

 UPDATE OPERATIONS 
// Step 4. Update one document
           db.students.updateOne(
                 { name: "Ravi" },
                 { $set: { age: 23 } }
               )
           db.students.find({ name: "Ravi" })

// Step 5: Update multiple documents
db.students.updateMany(
  { dept: "CSE" },
  { $set: { status: "Active" } }
)
db.students.find()
// step 6: Increment marks
db.students.updateOne(
  { name: "Anu" },
  { $inc: { marks: 5 } }
)

DELETE OPERATIONS 
// Step 7: Delete one document
	db.students.deleteOne({ name: "Meena" })

	db.students.find()

// Step 8: Delete multiple documents
		db.students.deleteMany({ dept: "CSE" })

// Step 9: Delete all documents
		db.students.deleteMany({})
`);
});

// Export app for testing or further usage
export default app;
