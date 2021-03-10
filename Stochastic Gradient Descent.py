#Import libraries
import numpy as np
from sklearn.datasets.samples_generator import make_regression 
import pylab

# Generate random data
x, y = make_regression(n_samples=100, n_features=1, n_informative=1, 
                    random_state=0, noise=35) 
m, n = np.shape(x)
x = np.c_[ np.ones(m), x] 

#Visualizing data
pylab.plot(x[:, 1],y, 'o')


# Initializing parameters and learning rate   
alpha = 0.01 # learning rate
theta = np.ones(2) # Initial values of parameters
    
    
#Define Stochastic Gradient Descent Algorithm to update parameters in the form of the function below:
def sgd(alpha, theta,  x, y, numIterations):
    m = x.shape[0] # number of samples
#    x_transpose = x.transpose()
    y_new = y.reshape((y.shape[0], 1))
    data = np.concatenate((x,y_new), axis=1)
    np.random.seed(42)
    np.random.shuffle(data)
    for iter in range(numIterations):
        for sample in range(m):
            hypothesis = np.dot(x[sample], theta) # matrix multiplication
            loss = hypothesis - y[sample] # or y - hypothesis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,                                                                   
            J = (1.0/2)*(loss ** 2)   # cost function
            print ("iter %s | J: %.3f" % (iter, J))  
#            gradient = np.dot(x_transpose, loss) / m   
            gradient = loss*x[sample]
            theta = theta - alpha * gradient  # update rule
#            print(theta)        
    return theta

# Applying stochastic gradient descent algorithm    
    theta_sgd = sgd(alpha, theta, x, y, 2)
    print (theta_sgd)


 # plot SGD
    for i in range(x.shape[1]):
        y_predict = theta_sgd[0] + theta_sgd[1]*x 
    pylab.plot(x[:,1],y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()
    print ("SGD Done!")
