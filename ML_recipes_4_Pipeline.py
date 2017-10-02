
# coding: utf-8

# In[2]:


#import a dataset
from sklearn import datasets
iris=datasets.load_iris()
X=iris.data
y=iris.target


# In[8]:


#Train-Test Split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.5)

#Decsion Tree
from sklearn import tree
my_classifier=tree.DecisionTreeClassifier()
my_classifier.fit(X_train,y_train)
predictions=my_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

#KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier=tree.DecisionTreeClassifier()
my_classifier.fit(X_train,y_train)
predictions=my_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


# In[ ]:




