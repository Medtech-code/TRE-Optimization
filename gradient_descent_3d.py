
# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt
from sympy import *
from sympy.vector import CoordSys3D, matrix_to_vector
from sympy.vector import Vector
N = CoordSys3D('N')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
import sympy.physics.vector as spv
from sympy.matrices.dense import matrix_multiply_elementwise as dot

# from sympy.physics.vector import *
# N = ReferenceFrame('N')
u,v,w = symbols('u v w')

x = np.linspace(-30, 30, 100)
y = np.linspace(-30, 30, 100)
N_fid=3
fids =Matrix([[10,0,0],[20,10,0],[15,0,25]])

target = Matrix([u,v,w])
target_vec = matrix_to_vector(target,N)
d_x_cross = target_vec.cross(N.i).to_matrix(N)
d_y_cross = target_vec.cross(N.j).to_matrix(N)
d_z_cross = target_vec.cross(N.k).to_matrix(N)

d_x = sum(dot( d_x_cross,d_x_cross ))
d_y = sum(dot( d_y_cross,d_y_cross ))
d_z = sum(dot( d_z_cross,d_z_cross ))

f_x = 0
f_y = 0
f_z = 0

for i in range(N_fid):
    vec = matrix_to_vector(fids[i,:],N)
    cross_x = vec.cross(N.i).to_matrix(N)
    cross_y = vec.cross(N.j).to_matrix(N)
    cross_z = vec.cross(N.k).to_matrix(N)

    f_x = f_x + sum(dot( cross_x,cross_x ))

    f_y  = f_y + sum(dot( cross_y,cross_y ))

    f_z  = f_z +  sum(dot( cross_z,cross_z ))

f_x = f_x/N_fid
f_y = f_y/N_fid
f_z = f_z/N_fid 
fle = 0.3
d_by_f = d_x/f_x + d_y/f_y + d_z/f_z

tre =sqrt( fle**2 *(1/N_fid + (d_by_f/3)))


grad_x = diff(tre,u)
grad_y = diff(tre,v)
grad_z = diff(tre,w)
  

X, Y = np.meshgrid(x, y)
def f(x, y):
    return x**2 + y**2
u,v = symbols('u v')
func = u**2+v**2

Z = u**2+v**2
global init_x , init_y,init_z

centro = np.mean(fids,axis=0)
init_x = centro[0]
init_y = centro[1]
init_z = centro[2]

x_g = centro[0]
y_g = centro[1]
z_g = centro[2]


# grad_x = diff(func,u)
# grad_y = diff(func,v)
# print(grad_x)

def unit(vector):
    """ Returns the unit vector of the vector.  """
    if np.linalg.norm(vector)== 0:
        return np.array([0,0,0])
    else:
        return vector / np.linalg.norm(vector)

# Define the function to update the plot
def update_plot(num_steps = 100):
    global X,Y,Z,init_x,init_y,init_z,x_g,y_g,z_g
    # Generate random 3D data
    scale=10
    alpha = 10
    beta = 0.1
    # update for gradient
    gradient_vec =np.array( [grad_x.subs([(u,init_x),(v,init_y),(w,init_z)]), grad_y.subs([(u,init_x),(v,init_y),(w,init_z)]),grad_z.subs([(u,init_x),(v,init_y),(w,init_z)])])
    
    init_x = init_x+gradient_vec[0]*alpha
    init_y = init_y+gradient_vec[1]*alpha
    init_z = init_z+gradient_vec[2]*alpha
    vec = np.array(np.cross(gradient_vec,np.array([1,0,0]))).astype(float)
    vec = vec/np.linalg.norm(vec)
    tangent_grad = unit(vec)
    x_g = x_g + tangent_grad[0]*beta  
    y_g = y_g + tangent_grad[1]*beta
    z_g = z_g + tangent_grad[2]*beta
    
    x = init_x
    y = init_y
    z = init_z
    

    # Clear the previous plot and update with new data
    # ax.clear()
    # ax = fig.gca(projection='3d')
   
    
    ax.scatter(x_g, y_g, z_g, c='y', marker='*',)
    ax.scatter(x, y, z, c='g', marker='*',)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # bx.clear()
    # bx.
    ax.set_title('Real-Time 3D Scatter Plot')

# Create a Matplotlib figure and 3D axes

# Set the initial view limits (optional)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
# bx = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-15, azim=15, roll=0)


landscape = f(X, Y)
blue = np.array([0., 0.5, 0.])
rgb = np.tile(blue, (landscape.shape[0], landscape.shape[1], 1))

ls = LightSource()
illuminated_surface = ls.shade_rgb(rgb, landscape)

# surf = ax.plot_surface(X, Y,landscape,linewidth=0,
#                 antialiased=False,
#                 facecolors=illuminated_surface,alpha=0.1)
# fig.colorbar(surf)
ax.scatter(fids[:,0],fids[:,1],fids[:,2],c='r',marker='*')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(-50, 50)
#multpile plots in the same figure matplot lib
# Create an animation using the FuncAnimation class
# The 'interval' parameter controls the update interval in milliseconds
plot=update_plot(1000)

ani = FuncAnimation(fig, update_plot, interval=10)  # Update every 1 second

# Display the animation
plt.show()

    

