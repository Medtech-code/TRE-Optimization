from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return np.where(x == 0, 0, np.sin(1/x))

def df(x):
    return np.where(x == 0, 0, np.cos(1/x) * (-x**2))
def diff(x):
    clim = 1
    return np.where(x == 0, 0, (f(x+0.5* clim)-f(x-0.5* clim))/clim)

def gradient_descent(initial_x, learning_rate, num_iterations):
    x = initial_x
    x_history = [x]

    # for i in range(num_iterations):`  `
    #     gradient = df(x)
    #     x = x - learning_rate * gradient
    #     x_history.append(x)
    while abs(df(x)) > 0.0001 :
        gradient = df(x)
        print(gradient,diff(x))
        x = x - learning_rate * gradient
        x_history.append(x)
    
    return x, x_history

initial_x = -0.20
learning_rate = 0.1
num_iterations = 100
x, x_history = gradient_descent(initial_x, learning_rate, num_iterations)
print("Local minimizer  and local minima{:.2f} {:.2f}".format(x,f(x)))
fig, ax = plt.subplots()
x_vals = np.linspace(-2, 2, 1000)
ax.plot(x_vals, f(x_vals))
line, = ax.plot([], [], 'rx')
local_min_line = ax.axvline(x,  color='g', linestyle='--', label='Local Minimum')

def init():
    line.set_data([], [])
    return line,
# update frame
def update(frame):
    x_history_frame = x_history[:frame]
    y_history_frame = f(np.array(x_history_frame))
    line.set_data(x_history_frame, y_history_frame)
    return line,

# Set plot labels and title
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Gradient Descent')
ax.legend()

# Create the animation
ani = FuncAnimation(fig, update, frames=num_iterations + 1, init_func=init, blit=True)
plt.show()


# fig.tight_layout()
# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(x_history), init_func=init, blit=True)
# plt.show()
