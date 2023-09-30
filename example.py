
import numpy as np
from minigrad import DeepFeedforwardNetwork 

# Example usage (XOR Problem)
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

"""
if use without opt:
nn = DeepFeedforwardNetwork(2, [3]*2+[1], learning_rate=0.8)
nn.train(x_train, y_train, epochs=5100)
"""

nn = DeepFeedforwardNetwork(2, [3]*2+[1], learning_rate=0.008) # Note that RAdam uses a diff lr, e.g. 0.001 
nn.train(x_train, y_train, epochs=2000) # less epochs 

y_pred = nn.pred(x_train)
print(y_pred) #[0,1,1,0]


#I'd like to add more complex functionalities, but I don't have time to write other optimizers, etc., at least right now.



