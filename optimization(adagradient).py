import numpy as np
from numpy import arange
from numpy.random import rand
from numpy import meshgrid
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import seed
# Define the function
def objective(x, y):
    # # Avoid division by zero
    # x_nonzero = np.where(x == 0, 1e-8, x)
    # return np.sin(1 / x_nonzero)
    return x**2.0+y**2.0

def derivative(x, y):
    # gradient_x = np.where(x == 0, 0, np.cos(1/x) * (-1/x**2))
    # gradient_y = np.zeros_like(gradient_x)  # Set gradient y component (dummy value)
    # return np.array([gradient_x, gradient_y])
    return np.asarray([x*2.0,y*2.0])

# Define the range for input
r_min, r_max = -1.0, 1.0

# Sample input range uniformly at 0.01 increments
xaxis = arange(r_min, r_max, 0.01)
yaxis = arange(r_min, r_max, 0.01)

# Create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)

# Compute targets
results = objective(x, y)

# Create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.add_subplot(111, projection='3d')

surface = axis.plot_surface(x, y, results, cmap='jet')

# Show the plot
pyplot.show()
# adagradient 
def adagrad(objective, derivative, bounds, number_of_iterations, step_size):
    solutions = []

    # the the boundary condition for choosing a random point within the bound
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # print("solution=",solution)
    # In the line of code sq_grad_sums = [0.0 for _ in range(bounds.shape[0])], the underscore (_) is used as a variable name to indicate that the value it holds is not going to be used in the loop
    sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
    
    for i in range(number_of_iterations):
        gradient = derivative(solution[0], solution[1])

    # the below loop runs 2 times as the gradient.shape[0] is 2
        for j in range(gradient.shape[0]):

    # this is used to calculate the sum of squared partial derivative for each input variable (solution)
            sq_grad_sums[j] += gradient[j] ** 2

        new_solution=[]
        for j in range(gradient.shape[0]):
    # the initial step size is set and the step-size/learning rate is calculated by initial step size divided by sqrt of sum of squared partial derivative,1e-8 is given to avoid zero divison
            learning_rate = step_size / (1e-8 + np.sqrt(sq_grad_sums[j]))
            # print("learning rate=",learning_rate)

    
            value=solution[j]- learning_rate * gradient[j]
            new_solution.append(value)
    # 
        solution=new_solution
        solutions.append(solution)
        solution_eval=objective(solution[0],solution[1])
        # print('>%d f(%s) = %5f' % (i,solution,solution_eval))
        # print("new_solution=",new_solution)
        print("solutions=",solutions)
        # print("solution=",solutions)
    return solutions

np.random.seed(1)
bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
number_of_iterations = 100
step_size = 0.1

solutions_adagrad = adagrad(objective, derivative, bounds, number_of_iterations, step_size)
print("solutions_adagrad=",solutions_adagrad)
xaxis = arange(bounds[0, 0], bounds[0, 1], 0.1)
yaxis = arange(bounds[1, 0], bounds[1, 1], 0.1)
x, y = meshgrid(xaxis, yaxis)
results = objective(x, y)

plt.contourf(x, y, results, levels=50, cmap='jet')
solutions = np.asarray(solutions_adagrad)
plt.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
local_minima = np.array(solutions)[np.argmin([solutions[:, 0], solutions[:, 1]]), :]
print("local minima=",local_minima)
# plt.plot(local_minima[0], local_minima[1], 'go')  # Mark local minimum with red circle
plt.show()

      
