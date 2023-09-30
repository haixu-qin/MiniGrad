import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def safe_sigmoid(x):
    x = np.clip(x, -500, 500)  # Clip values to avoid overflow
    return 1.0 / (1.0 + np.exp(-x)) #sigmoid(x)

def sigmoid_derivative(x):
    return x * (1 - x)

class DeepFeedforwardNetwork:
    def __init__(self, input_size, layer_sizes, learning_rate=0.01):
        self.layers = [] #w
        self.biases = [] #b
        prev_size = input_size
        for size in layer_sizes:
            self.layers.append(np.random.randn(prev_size, size) * 1)  # 0.01, small weight initialization
            self.biases.append(np.zeros((1, size)))
            prev_size = size
        self.learning_rate = learning_rate

    def forward(self, x):
        self.outputs = []
        self.inputs = [x]
        for w, b in zip(self.layers, self.biases):
            x = safe_sigmoid(np.dot(x, w) + b)
            self.inputs.append(x)
            self.outputs.append(x)
        return self.outputs[-1]

    def compute_loss(self, predicted, actual):
        return np.mean(0.5 * (predicted - actual) ** 2)  # MSE loss

    def backward(self, y_true):
        # Start with the gradient from the loss
        grad = (self.outputs[-1] - y_true)
        
        # Reverse iterate through layers
        for i in range(len(self.layers) - 1, -1, -1):
          
            pre_activation_grad = grad * sigmoid_derivative(self.outputs[i])
            weight_grad = np.dot(self.inputs[i].T, pre_activation_grad)
            
            # Update weights and biases
            self.layers[i] -= self.learning_rate * weight_grad
            self.biases[i] -= self.learning_rate * np.sum(pre_activation_grad, axis=0)
            
            grad = np.dot(pre_activation_grad, self.layers[i].T)

    def backward_opt(self, y_true):
        # Start with the gradient from the loss
        grad = (self.outputs[-1] - y_true)
        
        # RAdam hyperparameters
        beta1 = 0.9 # m coef
        beta2 = 0.999 # v coef 
        epsilon = 1e-8 # avoid div by zero 
        rho_inf = 2/(1-beta2)-1
        
        # If not initialized, initialize moving averages and timestep
        if not hasattr(self, 'm'):
            self.m = [np.zeros_like(layer) for layer in self.layers]
            self.v = [np.zeros_like(layer) for layer in self.layers]
            self.m_bias = [np.zeros_like(bias) for bias in self.biases]
            self.v_bias = [np.zeros_like(bias) for bias in self.biases]
            self.t = 0

        # Increment timestep for RAdam for each pass
        self.t += 1

        # Reverse iterate through layers
        for i in range(len(self.layers) - 1, -1, -1):

            # dL/dw. same as basic SGD 
            pre_activation_grad = grad * sigmoid_derivative(self.outputs[i])
            weight_grad = np.dot(self.inputs[i].T, pre_activation_grad)
            
            # RAdam computations for weights
            # 1. momentum and squared gradient (RMSprop)
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * weight_grad # momentum 
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * weight_grad**2 # squared gradient 

            # 2. bias corrected terms (Adam)
            m_corrected = self.m[i] / (1 - beta1**self.t)
            #v_corrected = self.v[i] / (1 - beta2**self.t)           

            # 3. rectified terms (RAdam)
            # Compute bias correction terms and rho_t for RAdam
            #delta_m = np.sqrt(1 - beta2**self.t)
            rho_t = rho_inf - (2 * self.t * beta2**self.t) / (1 - beta2**self.t)
            l_t = 1
            l_t_biases = 1
            r_t = 1
            if rho_t > 4:
              l_t = np.sqrt((1 - beta2**self.t)/(self.v[i] + epsilon))
              l_t_biases = np.sqrt((1 - beta2**self.t) / (self.v_bias[i] + epsilon))
              r_t = np.sqrt( ((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t) )
              
            # 4. update w 
            # Update weights using RAdam rule
            self.layers[i] -= self.learning_rate * m_corrected * r_t * l_t
            
            # RAdam computations for biases
            # similar process to w 
            bias_grad = np.sum(pre_activation_grad, axis=0)
            self.m_bias[i] = beta1 * self.m_bias[i] + (1 - beta1) * bias_grad
            self.v_bias[i] = beta2 * self.v_bias[i] + (1 - beta2) * bias_grad**2

            m_bias_corrected = self.m_bias[i] / (1 - beta1**self.t)
            #v_bias_corrected = self.v_bias[i] / (1 - beta2**self.t)

            # Update biases using RAdam rule
            self.biases[i] -= self.learning_rate * m_bias_corrected * r_t * l_t_biases
            
            grad = np.dot(pre_activation_grad, self.layers[i].T)
    
    def train(self, x, y, epochs):
        for epoch in range(epochs):
            predictions = self.forward(x)
            loss = self.compute_loss(predictions, y)
            self.backward_opt(y) #self.backward(y) if without opt 
            if epoch % 100 == 0:  # Print loss every 100 epochs
                print(f"Epoch {epoch}, Loss: {loss}")
                #print(predictions)
                print([round(pred[0]) for pred in predictions]) #debug

    def pred(self, x):
      predictions = self.forward(x)
      return [round(pred[0]) for pred in predictions]

    def get_weights(self):
      return self.layers 

    def get_biases(self):
      return self.biases





