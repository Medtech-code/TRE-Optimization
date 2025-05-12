import numpy as np
from numpy import arange
from numpy.random import rand
from numpy import meshgrid
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import seed
from scipy.misc import derivative as scipy_derivative
from matplotlib.animation import FuncAnimation
import math
import time
from sympy import symbols, diff
u, v = symbols('u v')
expression = u**2+ v**2
# Define the function
def objective(u, v):
    # creating an expression
    return u**2.0+v**2.0
  
def derivative(a, b):
    # gradient_x = np.where(x == 0, 0, np.cos(1/x) * (-1/x**2))
    # gradient_y = np.zeros_like(gradient_x)  # Set gradient y component (dummy value)
    # return np.array([gradient_x, gradient_y])
    # return np.asarray([x*2.0,y*2.0])
    derivative_x = diff(expression, u)
    derivative_y = diff(expression, v)
    gradient=np.array([derivative_x.evalf(subs={u:a,v:b}), derivative_y.evalf(subs={u:a,v:b})]) 
    print("gradient=",gradient)
    return gradient

def RMSProp(objective,derivative,bounds,number_of_iterations,step_size,rho):

    # generate an initial point
    solution=bounds[:,0]+rand(len(bounds))*(bounds[:,1]-bounds[:,0])

    # list of average squared gradient for each variable
    sq_grad_avg=np.zeros(bounds.shape[0])

    for i in range(number_of_iterations):
        gradient=derivative(solution[0],solution[1])    
    
    for j in range(gradient.shape[0]):
    # calculate the squared gradient and update the moving average of the squared gradient
        sq_grad=gradient[j]**2
        sq_grad_avg[j]=(sq_grad_avg[j]*rho)+((sq_grad)*(1.0-rho))
        

    
        

 
        