import numpy as np
import matplotlib.pyplot as plt
from TRE_functions import compute_ortho,rms,compute_targetdist,plot_fids
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
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tre_calc import compute_tre

import numpy as np
import matplotlib.pyplot as plt


def generate_points_inside_ellipse(a, b,c,count):
    
    # angle = 2 * np.pi * np.random.uniform()
    # scaling_factor = np.random.uniform()
    # the array contains 5 rows and 1 column
    theta = np.random.rand(count,1)*180
    phi = np.random.rand(count,1)*180 - np.pi/2
    # Calculate the x and y coordinates of the random point within the ellipse
    x =  a * np.sin(theta) * np.cos(phi)
    y =   b * np.sin(theta) * np.sin(phi)
    z =   c * np.cos(theta)

    


    return np.hstack((x,y,z))


def compute_TRE(fiducial,target):
    xc=np.mean(fiducial,axis=0)
    D=fiducial-xc

    axis = compute_ortho(D)
    target_dist = []
    for i in range(3):
        origin = xc
        line_pt1 = origin
        line_pt2 = origin + axis[i]
        point = target
        compute_targetdist(point,line_pt1,line_pt2)
        target_dist.append(compute_targetdist(point,line_pt1,line_pt2))
    # print(f'target distances :{target_dist}')
    datax = []
    datay = []
    dataz = []
    for i in range(3):
        for j in range(len(fiducial)):
            origin = xc
            line_pt1 = origin
            line_pt2 = origin+axis[i]
            point = fiducial[j]
            dist = compute_targetdist(point,line_pt1,line_pt2)
            if i == 0:
                datax.append(dist)      
            elif i == 1:
                datay.append(dist)
            elif i == 2:
                dataz.append(dist)
    # print(datax,datay,dataz)
    d = target_dist
    f = np.zeros((3,1))
    f[0]=rms(datax)
    f[1]=rms(datay)
    f[2]=rms(dataz)
    # print(f)
    N =len(fiducial)
    gamma = np.square(d[0]/rms(datax))+np.square(d[1]/rms(datay))+np.square(d[2]/rms(dataz))
    tre = np.sqrt((1+gamma/3)/N)
    return tre ,axis

if __name__ == "__main__":
    a = 10
    b = 10
    c= 10
    desired_tre=0.5
    # Numberof fiducials=5
    fiducials = generate_points_inside_ellipse(a, b,c,5)
    
    # fiducials = np.insert(fiducials, 2, 0, axis=1)
    target=np.array([0,0,0])
    # target = np.mean(fiducials,axis=0)
    # target=np.insert(target,2,0)
    # target = np.sum(fiducials[:3], axis=0) / 3
    # tre,axis = compute_TRE(fiducials, target)
    # print(tre)
    # generate 100 theta values in the range of 0 to 2pi
    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source]
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(-np.pi/2, np.pi/2, 100)

    theta, phi = np.meshgrid(theta, phi)
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    r = 25
    ax.set_xlim([-r,r])
    ax.set_ylim([-r,r])
    ax.set_zlim([-r,r])

    # Compute and plot the axes
    xc = np.sum(fiducials[:3], axis=0) / 3
    # axis = compute_unit(np.subtract(fiducials, xc))
    distance= 0.1

    # function for updating one fiducial position
    def update_fiducial_position(fiducials, alpha, num_steps):
        updated_fiducials = []  # List to store updated fiducials at each step
        tre = []


        for step in range(num_steps):
            current_fid_position = fiducials[0]
            tre_original,axis=compute_TRE(fiducials,target)
            distance_current=np.linalg.norm(current_fid_position-target)
            
            theta_ = np.linspace(0, 2 * np.pi, 4)
            phi_ = np.linspace(-np.pi/2, np.pi/2, 4)

            theta, phi = np.meshgrid(theta_, phi_)
            x = np.sin(theta) * np.cos(phi)
            y =  np.sin(theta) * np.sin(phi)
            z =  np.cos(theta)

            fid = np.vstack(([x.reshape(-1)],[y.reshape(-1)],[z.reshape(-1)])).transpose()
            # len of fiducials is 4
            for j  in range(len(fiducials)):
                # len of fid is 16
                for i in range(len(fid)):
                
                    current_fid_position = fiducials[j]
                    current_fid_position = current_fid_position + alpha *fid[i]
                    update_fid = fiducials.copy()
                    update_fid[j] = current_fid_position
                    updated_tre = compute_tre(update_fid, target)

                    current_tre = compute_tre(fiducials, target)
                    # print(updated_tre-current_tre)
                    if updated_tre < compute_tre(fiducials, target):
                        fiducials = update_fid
                        
                        tre.append(updated_tre)
                        updated_fiducials.append(fiducials)
                
        return np.array(updated_fiducials), tre 

            
        


    fiducial_points,tre_values=update_fiducial_position(fiducials,alpha=3,num_steps=50)
    # fiducial_points=np.array(fiducial_points).reshape(4,3)
    updated_fid_x=fiducial_points[:,:,0].flatten()
    updated_fid_y=fiducial_points[:,:,1].flatten()
    updated_fid_z=fiducial_points[:,:,2].flatten()

    # create the plot
    test_tre,axis = compute_TRE(fiducials, target)
                   

    for i in range(3):
        
        ax.quiver(xc[0], xc[1], xc[2], a*axis[i, 0], a*axis[i, 1], a*axis[i, 2], color='purple', linestyle='--', label=f'Axis {i+1}')

    ax.scatter(updated_fid_x, updated_fid_y, updated_fid_z, color='green', marker='x')
    ax.scatter(target[0], target[1], target[2], color='red', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'TRE: {test_tre:.2f} | 4 Random Points Inside an Ellipsoid')
    plt.grid(True)
    plt.show(block=True)
    # plt.pause(5)


def init():
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(-np.pi/2, np.pi/2, 100)

    theta, phi = np.meshgrid(theta, phi)
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    line,=ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    r = 25
    ax.set_xlim([-r,r])
    ax.set_ylim([-r,r])
    ax.set_zlim([-r,r])

    return fig,line



def animate(frame):

    x_values.append(updated_fid_x[frame])
    y_values.append(updated_fid_y[frame])
    z_values.append(updated_fid_z[frame])
    line.set_data(x_values,y_values,z_values)
    plt.cla()
    plt.plot(x_values,y_values,z_values)

x_values=[]
y_values=[]
z_values=[]

fig,line=init()
ani=FuncAnimation(fig,animate,frames=len(tre_values),init_func=init,blit=True)
plt.show


    


    

        
