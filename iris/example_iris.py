
from sklearn import datasets 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#we use scikit-learn bc we want to use the datasets. we could write them but it's not for the purpose of this tutorial.

from minigrad_iris import DeepFeedforwardNetwork 

# Load the Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Preprocess
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

encoder = OneHotEncoder(sparse_output=False).fit(y)
y_onehot = encoder.transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

print(X_train[1], len(X_train[1]), y_train[1], len(y_train[1]))


#train the model
input_size = len(X_train[1])
num_nodes = 5
num_layers = 3
output_size = len(y_train[1])
layer_sizes = [num_nodes]*num_layers + [output_size]
nn = DeepFeedforwardNetwork(input_size, layer_sizes, learning_rate=0.008) # Note that RAdam uses a diff lr, e.g. 0.001 
nn.train(X_train, y_train, epochs=800) #more epochs for larger datasets 
#epochs=2000 without opt

y_pred = nn.pred(X_test)
mse, accuracy = nn.compare(y_pred, y_test)
print(mse, accuracy) #best performance: 0.0001, 1.0. time: <1s. 
#Avg. performance: 0.0112, 0.9667. which is good enough. :).







