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


# import sympy
from sympy import *
u, v = symbols('u v')
expression = u**2+ v**2
# Define the function
def objective(u, v):
    # creating an expression
    return u**2.0+v**2.0
  
def derivative(a, b):
    # gradient_x = np.where(x == 0, 0, np.cos(1/x) * (-1/x**2))
    # gradient_y = np.zeros_like(gradient_x)  # Set gradient y component (dummy value)+
    # return np.array([gradient_x, gradient_y])
    # return np.asarray([x*2.0,y*2.0])
    derivative_x = diff(expression, u)
    derivative_y = diff(expression, v)
    gradient=np.array([derivative_x.evalf(subs={u:a,v:b}), derivative_y.evalf(subs={u:a,v:b})]) 
    print("gradient=",gradient)
    return gradient
    
# adagradient 
def adagrad(objective, derivative, bounds, number_of_iterations, step_size):
    solutions = []
    
    # the the boundary condition for choosing a random point within the bound
    solution = np.zeros(2)
    solution[0] = bounds[0, 0] + (bounds[0, 1] - bounds[0, 0])
    solution[1] = bounds[1, 0] + (bounds[1, 1] - bounds[1, 0]) 
    # solution=[100,100]

    # print("solution=",solution)
    # In the line of code sq_grad_sums = [0.0 for _ in range(bounds.shape[0])], the underscore (_) is used as a variable name to indicate that the value it holds is not going to be used in the loop
    sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
    local_minima_reached = False  # A flag to indicate if local minima is reached
    local_minima_coords = None
    for i in range(number_of_iterations):
        gradient = derivative(solution[0], solution[1])

    # the below loop runs 2 times as the gradient.shape[0] is 2
        for j in range(gradient.shape[0]):

    # this is used to calculate the sum of squared partial derivative for each input variable (solution)
            sq_grad_sums[j] += gradient[j] ** 2

        new_solution=[]
        diff_small = True 
        gradient_norm = math.sqrt(np.sum(gradient**2))  # Calculate the norm of the gradient
        if gradient_norm < 1e-6:
            diff_small = True

        for j in range(gradient.shape[0]):
            learning_rate = step_size / (1e-8 + math.sqrt(sq_grad_sums[j]))
            value = solution[j] - learning_rate * gradient[j]
            new_solution.append(value)
            diff = abs(value - solution[j])

            if diff >= 0.001:
                diff_small = False

        solution = new_solution
        solutions.append(solution)

        if diff_small:
            local_minima_reached = True
            local_minima_coords = solution.copy()
            break

        

        

    return solutions, local_minima_coords
        # print("new_solution=",new_solution)
        # print("solutions=",solutions)
        # print("solution=",solutions)

    
np.random.seed(1)
bounds = np.asarray([[-100, 100], [-100, 100]])
# print("range=",len(range))
# print("solutions_adagrad=",solutions_adagrad)
xaxis = arange(bounds[0, 0], bounds[0, 1], 0.1)
yaxis = arange(bounds[1, 0], bounds[1, 1], 0.1)
# yaxis = [1,2]
x, y = meshgrid(xaxis, yaxis)
results = objective(x, y)
# print("results=",results)
step_size = 8
number_of_iterations = len(results)
solutions_adagrad,local_minima = adagrad(objective, derivative, bounds, number_of_iterations, step_size)
if local_minima is not None:
    print("Local Minima Coordinates:", local_minima)
else:
    print("Local Minima Not Found")
# Create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.add_subplot(111, projection='3d')
surface = axis.plot_surface(x, y, results, cmap='jet',alpha=0.2)
solutions = np.asarray(solutions_adagrad)
# print("solutions=",solutions)

# axis.scatter(local_minima[0], local_minima[1], c='red', marker='x', s=100)

def update_frame(frame, solutions, objective_func, local_minima_reached, local_minima):
        
    point = solutions[frame]
    value_objective = objective_func(point[0], point[1])
    scatter= axis.scatter([point[0]], [point[1]], [value_objective], color='lime', marker='o', s=10)
    scatter.set_offsets(np.array([point]))
    scatter._sizes = [50]
    axis.scatter(point[0], point[1], value_objective)

    if local_minima_reached[0]:
        return axis.scatter,

    if local_minima is not None:  # Check if local_minima is available
        if abs(objective_func(local_minima[0], local_minima[1]) - value_objective) < 0.001:
            local_minima_reached[0] = True
            axis.scatter(local_minima[0], local_minima[1], c='r', marker='+', s=100)
         
    return scatter,


# Plot the initial point
# initial_point = solutions[0]
# initial_value = objective(initial_point[0], initial_point[1])
# scatter = axis.scatter([initial_point[0]], [initial_point[1]], [initial_value], color='lime', marker='o', s=10)

local_minima_reached = [False]  # A list to store whether local minima is reached

# Create the animation
animation = FuncAnimation(
    figure,
    update_frame,
    frames=range(1, len(solutions)),  # Exclude the initial point
    fargs=(solutions, objective, local_minima_reached, local_minima),
    blit=True,
    repeat=False
)
# Label the axes
axis.set_xlabel('X')
axis.set_ylabel('Y')
axis.set_zlabel('Z')
# Add a title
plt.title('Surface Plot Animation with Local Minima')
# Show the animation
plt.show()



