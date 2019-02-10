
# coding: utf-8

# In[17]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import h5py


# In[18]:


with h5py.File('C:/Users/Surabhi/Desktop/IIITD/5th SEM/ML/Assignments/HW3_NN/data/Q1/MNIST_Subset.h5') as data:
    X = data['X'][:]
    Y = data['Y'][:]


# In[19]:


X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(X, Y)


# In[21]:


c = MLPClassifier(activation = 'logistic', solver = 'sgd', alpha = 0.001, batch_size = 20, max_iter = 100)
c.fit(x_train, y_train)


# In[22]:


predictions = c.predict(x_test)
print(accuracy_score(predictions, y_test))


# In[23]:


d = MLPClassifier(activation = 'logistic', solver = 'sgd', alpha = 0.001, hidden_layer_sizes=(100,50,50), batch_size = 20, max_iter = 100)
d.fit(x_train, y_train)


# In[24]:


predictions = d.predict(x_test)
print(accuracy_score(predictions, y_test))

