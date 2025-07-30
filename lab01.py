# creating a dataset for random values 
from sklearn.datasets import make_classification
import numpy as np 

X,y = make_classification(n_samples=100 , n_features=2 , n_informative=1 , n_redundant=0 , n_classes=2 ,n_clusters_per_class=1 , class_sep=10 , hypercube=False ,random_state=41 )

print(X)

#plotting the values 
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter')
plt.title("Classification Scatter Plot",fontsize =16)
plt.xlabel("Feature1",fontsize=14)
plt.ylabel("Feature2",fontsize=14)
plt.show()

#splitting dataset into training and testing 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.33,random_state=41)

#plotting training set
plt.figure(figsize=(8,5))
plt.scatter(X_train[:,0],X_train[:,1],c=y_train , cmap='winter')
plt.title("Training Set Plot",fontsize =16)
plt.xlabel("Feature1",fontsize=14)
plt.ylabel("Feature2",fontsize=14)
plt.show()

#plotting testing set
plt.figure(figsize=(8,5))
plt.scatter(X_test[:,0],X_test[:,1],c=y_test , cmap='winter')
plt.title("Testing Set Plot",fontsize =16)
plt.xlabel("Feature1",fontsize=14)
plt.ylabel("Feature2",fontsize=14)
plt.show()

#binary step function as activation function
def step(z):
    return 1 if z>=0 else 0

#perceptron 
def perceptron_training(X,y):
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1]) #initalizing all weights with 1 
    lr = 0.1 

    for epoch in range(2):
        errors = 0 
        for i in range(X.shape[0]):
            weighted_sum = np.dot(weights,X[i])
            y_hat = step(weighted_sum)
            err = y[i] - y_hat
            if err!=0:
                errors+=1
            weights = weights+ lr*(err)*X[i]
        print(f"Epoch{epoch+1}:Number of misclassifications = {errors}")
    return weights

final_weights = perceptron_training(X_train,y_train)
print(final_weights)

