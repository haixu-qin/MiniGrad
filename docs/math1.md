
## Math Explanation
\- Detailed Math Explanation for Backpropagation (`.backward()`):

### Ref:
You could read more about it [there](https://www.nature.com/articles/323533a0).

### Prerequisites:
basic knowledge of Calculus (derivative and chain rule) and Linear Algebra (matrix multiplication).

### Defs:
x = inputs \
w = weights \
b = bias 

z = sum(w*x) + b &nbsp;&nbsp;&nbsp;&nbsp; //combination  function \
y = sigmoid(z) (actual y) &nbsp;&nbsp;&nbsp;&nbsp; //activation function \
y^ = expected y &nbsp;&nbsp;&nbsp;&nbsp; //by convention y^ is the actual y, but we use it for expected y here so it simplifies the equations later

L = Loss (/ Total Error Function) = 1/2*sum((y^ - y)^2) &nbsp;&nbsp;&nbsp;&nbsp; //because one loss = (y^ - y), square it because of non-negativity and differentiability, and divide it by half because of mathematical convenience (that dL/dy = (y^ - y))

a = learning rate \
and delta_w = a * (- dL/dw) &nbsp;&nbsp;&nbsp;&nbsp; //because if the gradient is positive it means we want to decrease the w, and vice versa (based on the L graph). \
new_w = w + delta_w \
w += delta_w \
so w -= a * dL/dw

### Code 
The first line of the code \
`grad = (self.outputs[-1] - y_true)` \
grad is prev_grad dL/dy = (y - y^), which is the prev_grad for the weights for the last layer

Note that I used a variable (a node) instead a matrix (several nodes) in the equations, so that it's easier to understand.

Then \
`for i in range(len(self.layers) - 1, -1, -1):` \
start to loop backwards

#### Now we want to find the Gradient dL/dw, so that we could know what delta_w is

we can't directly compute it because there're other variables involved 

so we use chain rules

dL/dw = dL/dy * dy/dw

dL/dy = prev_grad

dy/dw = dy/dz * dz/dw (another chain rule)

\
dy/dz = d_sigmoid(z)

dz/dw = x

\
therefore, \
dL/dw = dL/dy * dy/dw \
= prev_grad * dy/dz * dz/dw \
= prev_grad * d_sigmoid(z) * x

which is

`pre_activation_grad = grad * sigmoid_derivative(self.outputs[i]) #prev_grad * d_sigmoid(z)`

`weight_grad = np.dot(self.inputs[i].T, pre_activation_grad) #prev_grad * d_sigmoid(z) * x`

#### Now we have dL/dw for this layer

so we update the weights and biases

w -= a * dL/dw \
`self.layers[i] -= self.learning_rate * weight_grad`

what about b? \
dL/db = dL/dz * dz/db (by chain rule) \
and dz/db = 1 \
so dL/db = dL/dz \
delta_b = -dL/dz (explanations for dL/dz below)

b -= a * dL/dz \
`self.biases[i] -= self.learning_rate * np.sum(pre_activation_grad, axis=0)`

so the gradient calculation for this layer is completed. 


#### Lastly, we want to compute the prev_grad dL/dy<sub>i</sub> for the next layer<sub>i-1</sub> (so we could use it for dL/dz<sub>i-1</sub>)

dL/dy<sub>i</sub> = dL/dz * dz/dy<sub>i</sub> (by chain rule)

dL/dz = dL/dy<sub>i+1</sub> * dy<sub>i+1</sub>/dz (by chain rule)

dz/dy<sub>i</sub> = w &nbsp;&nbsp;&nbsp;&nbsp; //because z = w*y<sub>i</sub> + b, y<sub>i</sub> is x for the curr layer 

\
dL/dy<sub>i+1</sub> = prev_grad

dy<sub>i+1</sub>/dz = d_sigmoid(z)

dL/dz = prev_grad * d_sigmoid(z) = pre_activation_grad (which is already computed)

\
so
dL/dy<sub>i</sub> = dL/dz * dz/dy<sub>i</sub> \
= pre_activation_grad * w

that is 

`grad = np.dot(pre_activation_grad, self.layers[i].T) #dot product of the pre_activation_grad vector and the transpose of the i_th vector of W` 

and it's done.

\
After the loop, that's one pass of backpropagation. \
If you want another pass, then you do another forward pass and re compute the loss, etc..

