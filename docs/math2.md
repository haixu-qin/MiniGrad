
## Math Explanation 
\- Math Explanation for Backpropagation with optimizer (RAdam) (`.backward_opt()`):

### Prerequisites:
You may want to read the Math Explanation for `.backward()`. \
It could be found [there](docs/math1.md).


## RAdam (Rectified Adam) Optimizer
You could read more about it [there](https://arxiv.org/abs/1908.03265). \
Particularly, "Algorithm 2: Rectified Adam." on p.g. 5.

RAdam is an optimization algorithm that rectifies the variance of the adaptive learning rate in the Adam optimizer.

## Intro

The primary motivation behind RAdam is to address the issue of slow convergence or divergence at the beginning of training due to large variances in adaptive learning rates.

## Mathematical Formulation

### Definitions
a - Learning rate \
g<sub>t</sub> - Gradient at time step t

β<sub>1</sub>, β<sub>2</sub> - Exponential decay rates for moving averages \
ε - A small constant to prevent division by zero (usually 10<sup>-8</sup>)  

m<sub>t</sub> - Moving average of the gradient (momentum) \
v<sub>t</sub> - Moving average of the squared gradient (RMSprop)

m̂<sub>t</sub> - Bias-corrected moving average

ρ<sub>∞</sub> - the maximum length of the approximated SMA (the simple moving average) \
ρ<sub>t</sub> - the length of the approximated SMA \
l<sub>t</sub> - Adaptive learning rate \
r<sub>t</sub> - Term to rectify the variance of adaptive learning rate (the variance rectification term)

### Code 

#### 1. Update Rules of Momentum and Squared Gradient (RMSprop)

Moving averages are updated as:  
m<sub>t</sub> = β<sub>1</sub> ⋅ m<sub>t-1</sub> + (1 - β<sub>1</sub>) ⋅ g<sub>t</sub> \
v<sub>t</sub> = β<sub>2</sub> ⋅ v<sub>t-1</sub> + (1 - β<sub>2</sub>) ⋅ g<sub>t</sub><sup>2</sup> 

```
self.m[i] = beta1 * self.m[i] + (1 - beta1) * weight_grad # momentum 
self.v[i] = beta2 * self.v[i] + (1 - beta2) * weight_grad**2 # squared gradient
```

and

```
self.m_bias[i] = beta1 * self.m_bias[i] + (1 - beta1) * bias_grad
self.v_bias[i] = beta2 * self.v_bias[i] + (1 - beta2) * bias_grad**2
```

Note that the momentum term for the i<sup>th</sup> layer is updated based on its previous value (at time t-1) and the current gradient, not the value for the prev (i+1)<sup>th</sup> layer.

#### 2. Bias Correction

Bias-corrected estimates are given by:  
m̂<sub>t</sub> = m<sub>t</sub> / (1 - β<sub>1</sub><sup>t</sup>)  

```
m_corrected = self.m[i] / (1 - beta1**self.t)
```

and

```
m_bias_corrected = self.m_bias[i] / (1 - beta1**self.t)
```

#### 3. Rectification Term

ρ<sub>t</sub> = ρ<sub>∞</sub> - 2 × t × β<sub>2</sub><sup>t</sup> / (1 - β<sub>2</sub><sup>t</sup>)

where ρ<sub>∞</sub> = 2 / (1 - β<sub>2</sub>) - 1.

If ρ<sub>t</sub> > 4:

l<sub>t</sub> = √((1 - β<sub>2</sub><sup>t</sup>) / (v<sub>t</sub> + ε))

l_bias<sub>t</sub> = √((1 - β<sub>2</sub><sup>t</sup>) / (v_bias<sub>t</sub> + ε))

r<sub>t</sub> = √( ((ρ<sub>t</sub> - 4) × (ρ<sub>t</sub> - 2) × ρ<sub>∞</sub>) / ((ρ<sub>∞</sub> - 4) × (ρ<sub>∞</sub> - 2) × ρ<sub>t</sub>) )

Otherwise, l<sub>t</sub>, l_bias<sub>t</sub>, and r<sub>t</sub> are 1s.

that is

```
rho_inf = 2/(1-beta2)-1
```

and

```
            rho_t = rho_inf - (2 * self.t * beta2**self.t) / (1 - beta2**self.t)
            l_t = 1
            l_t_biases = 1
            r_t = 1
            if rho_t > 4:
              l_t = np.sqrt((1 - beta2**self.t)/(self.v[i] * epsilon))
              l_t_biases = np.sqrt((1 - beta2**self.t) / (self.v_bias[i] * epsilon))
              r_t = np.sqrt( ((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t) )
```

#### 4. Parameter Update of W and Bias

Parameters are updated using:  
θ<sub>t</sub> = θ<sub>t-1</sub> - a ⋅ m̂<sub>t</sub> ⋅ r<sub>t</sub> ⋅ l<sub>t</sub>

```
self.layers[i] -= self.learning_rate * m_corrected * r_t * l_t
````

and

```
self.biases[i] -= self.learning_rate * m_bias_corrected * r_t * l_t_biases
```

#### Rest 
Rest are explained by comments.


## Lastly 
Note that [PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html)'s approach of RAdam is slightly different from the paper. This approach strictly follows the paper.


