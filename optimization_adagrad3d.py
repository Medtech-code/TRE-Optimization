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
u, v ,w = symbols('u v w')
expression = u**2+ v**2+w**2
# Define the function
def objective(u, v,w):
    # creating an expression
    return u**2.0+v**2.0+w**2
  
def derivative(a, b,c):
    # gradient_x = np.where(x == 0, 0, np.cos(1/x) * (-1/x**2))
    # gradient_y = np.zeros_like(gradient_x)  # Set gradient y component (dummy value)
    # return np.array([gradient_x, gradient_y])
    # return np.asarray([x*2.0,y*2.0])
    derivative_x = diff(expression, u)
    derivative_y = diff(expression, v)
    derivative_z = diff(expression,w)
    gradient=np.array([derivative_x.evalf(subs={u:a,v:b,w:c}), derivative_y.evalf(subs={u:a,v:b,w:c}),derivative_z.evalf(subs={u:a,v:b,w:c})]) 
    print("gradient=",gradient)
    return gradient
    
# adagradient 
def adagrad(objective, derivative, bounds, number_of_iterations, step_size):
    solutions = []

    # the the boundary condition for choosing a random point within the bound
    solution = np.zeros(3)
    solution[0] = bounds[0, 0] + (bounds[0, 1] - bounds[0, 0])
    solution[1] = bounds[1, 0] + (bounds[1, 1] - bounds[1, 0]) 
    solution[2] = bounds[2, 0] + (bounds[2, 1] - bounds[2, 0])
    # solution=[100,100]

    # print("solution=",solution)
    # In the line of code sq_grad_sums = [0.0 for _ in range(bounds.shape[0])], the underscore (_) is used as a variable name to indicate that the value it holds is not going to be used in the loop
    sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
    
    for i in range(number_of_iterations):
        gradient = derivative(solution[0], solution[1], solution[2])

    # the below loop runs 2 times as the gradient.shape[0] is 2
        for j in range(gradient.shape[0]):

    # this is used to calculate the sum of squared partial derivative for each input variable (solution)
            sq_grad_sums[j] += gradient[j] ** 2

        new_solution=[]
        for j in range(gradient.shape[0]):
    # the initial step size is set and the step-size/learning rate is calculated by initial step size divided by sqrt of sum of squared partial derivative,1e-8 is given to avoid zero divison
            learning_rate = step_size / (1e-8 + math.sqrt(sq_grad_sums[j]))
            # print("learning rate=",learning_rate)

    # the updated solution with different learning rate
            value=solution[j]- learning_rate * gradient[j]
            new_solution.append(value)
  
        solution=new_solution
        solutions.append(solution)
        # print("new_solution=",new_solution)
        # print("solutions=",solutions)
        # print("solution=",solutions)

        if local_minima_reached[0]:
        # Set the number of iterations to the step at which local minimum is reached
            number_of_iterations = i + 1

        # Exit the loop if local minimum is reached
            break 

    
    return solutions

np.random.seed(1)
bounds = np.asarray([[-10, 10], [-10, 10],[-10,10]])
number_of_iterations = 1
step_size = 10
local_minima_reached = [False]
solutions_adagrad = adagrad(objective, derivative, bounds, number_of_iterations, step_size)
# print("solutions_adagrad=",solutions_adagrad)
xaxis = arange(bounds[0, 0], bounds[0, 1], 0.1)
yaxis = arange(bounds[1, 0], bounds[1, 1], 0.1)
zaxis = np.arange(bounds[2, 0], bounds[2, 1], 0.1) 
# yaxis = [1,2]
x, y,z = meshgrid(xaxis, yaxis,zaxis)
results = objective(x, y,z)

# Create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.add_subplot(111, projection='3d')

surface = axis.plot_surface(x, y, results, cmap='jet',alpha=0.2)
solutions = np.asarray(solutions_adagrad)
# plt.plot(solutions[:, 0], solutions[:, 1], 'go')
local_minima = np.array(solutions)[np.argmin([solutions[:, 0], solutions[:, 1]]), :]
print("local minima=",local_minima)
# plt.plot(local_minima[0], local_minima[1], 'ro')  # Mark local minimum with red circle


def update_frame(frame, solutions, objective_func, scatter, local_minima_reached):
    if local_minima_reached[0]:
        return scatter,

    point = solutions[frame]
    value_objective = objective_func(point[0], point[1])
    scatter.set_offsets(np.array([point]))
    scatter._sizes = [50] 
    axis.scatter(point[0],point[1],value_objective)

    # Check if the difference in objective values is very small to consider it as reaching the local minimum
    if frame > 0 and abs(objective_func(solutions[frame-1][0], solutions[frame-1][1]) - value_objective) < 0.001:
        local_minima_reached[0] = True
        axis.scatter(point[0], point[1], c='red', marker='x', s=100)
   
    return scatter,

    


# Plot the initial point
initial_point = solutions[0]
initial_value = objective(initial_point[0], initial_point[1])
scatter = axis.scatter([initial_point[0]], [initial_point[1]], [initial_value], color='lime', marker='o', s=10)

local_minima_reached = [False]  # A list to store whether local minima is reached

# Create the animation
animation = FuncAnimation(
    figure,
    update_frame,
    frames=range(1, len(solutions)),  # Exclude the initial point
    fargs=(solutions, objective, scatter, local_minima_reached),
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