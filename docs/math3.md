
## Math Explanation 
\- Math Explanation for 1-D y (`example.py` and `minigrad.py`) vs. multi-D y (`example_iris.py` and `minigrad_iris.py` in `/iris`).

### forward pass (`.forward_one_y()` and `.forward_multi_y()`):
one y: sigmoid (y = w*sigmoid(x) + b). \
multi y: we use softmax for the last layer and sigmoid for all other layers.

### backward pass (`.backward()` or `.backward_opt()`):
pre_activation_grad = dL/dy<sub>i+1</sub> * actv_func' (derivations [there](./math1.md))

For the last layer.

one y: \
L = Total Error Function (Some people use Binary Cross Entropy for Binary Classification, that's another story.) \
dL/dy = (y^ - y)

actv_func = sigmoid \
actv_func' = d_sigmoid(z)

so pre_activation_grad = (y^ - y) * d_sigmoid(z) 

multi y:
L = Categorical Cross Entropy \
dL/dy = ...

actv_func = softmax (for the last layer) \
actv_func' = d_softmax(z)

so pre_activation_grad (dL(CCE)/dz) = dL/dy * d_softmax(z) = (y^ - y) \
(I won't go into the details of the derivations but if you're interested you could find it online.)

For other layers, they're the same.

that's why

```
        # Start with the gradient from the loss
        grad = (self.outputs[-1] - y_true)
        
        # Reverse iterate through layers
        for i in range(len(self.layers) - 1, -1, -1):
          
            pre_activation_grad = grad * sigmoid_derivative(self.outputs[i])
            # last layer for multi y 
            if i == len(self.layers) - 1 and y_true.shape[1] > 1:
              pre_activation_grad = grad
```

### compute loss (`.compute_loss_mse()` and `.compute_loss_cce()`):
Note that this is for the model evaluation, Not the computation (gradients). \
one y: MSE (mean squared error, or mean of TEF). \
multi y: CCE.













