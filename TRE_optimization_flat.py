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
# import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
# from tre_calc import compute_tre
from random import sample
# from functions import 
from functions_1 import compute_tre, min_dist_and_incr , compute_intermarker


tracking_tool=np.array([[0, -50	,0],
                            [0	,110	,0],
                            [0	,0	,0],
                            [-39,	90	,0],
                            [39,	60	,0]])

awl = np.array([[-47,   0,   0,],
                [-47,  75,   0],
                [ 58,   0,   0],
                [ 58, 120,   0]])

target=np.array([0,-140,65])
tre=compute_tre(awl,target)
print(tre)
print("Done")

# npoints=searchsample=12
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    # vector is divided by the norm of the vector to abtain unit vector
    vec /= np.linalg.norm(vec, axis=0)
    return vec.transpose()
def spherical_uniform_samples(search_sample,alpha=1):
    theta_ = np.linspace(0, 2 * np.pi, search_sample)
    phi_ = np.linspace(-np.pi/2, np.pi/2, search_sample)
    

    theta, phi = np.meshgrid(theta_, phi_)
    x = alpha * np.sin(theta) * np.cos(phi)
    y = alpha * np.sin(theta) * np.sin(phi)
    z = alpha * np.cos(theta)
    # z = np.zeros(theta.shape)
    fid = np.vstack(([x.reshape(-1)],[y.reshape(-1)],[z.reshape(-1)])).transpose()
    return fid
def circle_uniform_samples(search_sample):
    theta_ = np.linspace(0, 2 * np.pi, search_sample)
    x = np.cos(theta_).reshape(search_sample,1)
    y  = np.sin(theta_).reshape(search_sample,1)
    z = np.zeros(len(theta_)).reshape(search_sample,1)
    fid = np.hstack((x,y,z))
    return fid


if __name__ == "__main__":
    a = 135
    b = 105
    c= 0
    alpha = 5
    search_sample = 12
    desired_tre = 2
    cound_fid = 4
    noOfSteps = 200
    offset = 55
    target=np.array([-220,0,offset])
# def create_mesh(a,b):
    xx = np.linspace(0, a, 9)
    yy = np.linspace(-b/2, b/2, 9)
    theta, phi = np.meshgrid(xx, yy)
    
    x = theta
    y = phi
    z = np.zeros(theta.shape)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    r = a
    ax.set_xlim([0,2*r])
    ax.set_ylim([-r,r])
    ax.set_zlim([-r,r])
    mesh = np.vstack((x.reshape(-1),y.reshape(-1),z.reshape(-1))).transpose()
    sz = len(x.reshape(-1))
    
    l = sample(range(0, sz), cound_fid)
    fiducials = mesh[l,:]
    while min_dist_and_incr(compute_intermarker(fiducials),50,5)==False:
        l = sample(range(0, sz), cound_fid)
        fiducials = mesh[l,:]
        
    
    # fiducials = np.array()
    height  = np.array([0,a])
    width = np.array([-b/2,b/2])
    # ax.scatter(fiducials[:,0],fiducials[:,1],fiducials[:,2],color="green",marker="x")
    # plt.show()
    corners_x,corners_y = np.meshgrid(height/3,width/3)
    corners_x = corners_x.reshape(-1)
    corners_y = corners_y.reshape(-1)
    # fiducials = np.vstack((corners_x,corners_y,np.zeros(len(corners_x)))).transpose()
    


    # Compute and plot the axes
    xc = np.mean(fiducials, axis=0) 
    # axis = compute_unit(np.subtract(fiducials, xc))
    # distance= 0.1
    def checkConnstratints(fiducials):
        return  np.sum(fiducials[:,0]<=a)==len(fiducials) and np.sum(fiducials[:,0]>=0)==len(fiducials) and np.sum(np.abs(fiducials[:,1])<=b/2)==len(fiducials) and min_dist_and_incr(compute_intermarker(fiducials),50,5)
        # return 1

    # function for updating one fiducial position
    def update_fiducial_position(fid, alpha, num_steps):
        fiducials = fid.copy()
        updated_fiducials = []  # List to store updated fiducials at each step
        tre = []
        step_count = 0
        temp_fiducials = [] # temporary

        # alternative method for sampling the spherical values.


        
        # fid = sample_spherical(search_sample)
        fid = circle_uniform_samples(search_sample)


        while step_count < num_steps:


            step_count = step_count + 1
            for j  in range(len(fiducials)):
                min_tre = 10
                for i in range(len(fid)):
                
                    current_fid_position = fiducials[j]
                    current_fid_position = current_fid_position - alpha *fid[i]
                    update_fid = fiducials.copy()
                    update_fid[j] = current_fid_position
                    updated_tre = compute_tre(update_fid, target)
                    current_tre = compute_tre(fiducials, target)

                    # print(updated_tre-current_tre)
                    

                    
                    if updated_tre < min_tre  and checkConnstratints(update_fid):
                        temp_fiducials = update_fid
                        print("tempfid=",temp_fiducials)

                        min_tre = updated_tre
                        print(min_tre)
                        
                        
                if len(temp_fiducials)!=0:
                    if len(updated_fiducials)==0 :
                        updated_fiducials = temp_fiducials
                    else:
                        updated_fiducials = np.vstack((updated_fiducials,temp_fiducials))
                    fiducials = temp_fiducials
                
        return np.array(updated_fiducials), fiducials

            
        
    
    
    fiducial_points,result_fiducials=update_fiducial_position(fiducials,alpha=alpha,num_steps=noOfSteps)
    # fiducial_points=np.array(fiducial_points).reshpe(4,3)
    if len(fiducial_points) != 0:

        updated_fid_x=fiducial_points[:,0].flatten()
        updated_fid_y=fiducial_points[:,1].flatten()
        updated_fid_z=fiducial_points[:,2].flatten()

        # create the plot
        test_tre,axis = compute_tre(fiducials, target)
                    

        for i in range(3):
            
            ax.quiver(xc[0], xc[1], xc[2], a*axis[i, 0], a*axis[i, 1], a*axis[i, 2], color='purple', linestyle='--', label=f'Axis {i+1}')

        ax.scatter(updated_fid_x[:-1], updated_fid_y[:-1], updated_fid_z[:-1], color='blue', marker='x',alpha=0.01)
        ax.scatter(target[0], target[1], target[2], color='red', marker='x')
        ax.scatter(result_fiducials[:,0], result_fiducials[:,1], result_fiducials[:,2], color='green', marker='x')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_title(f'TRE: {test_tre:.2f} | 4 Random Points Inside an Ellipsoid')
        plt.grid(True)
        plt.show(block=True)
        # plt.pause(5)
        print(compute_tre(result_fiducials,target))
        print(result_fiducials)
    else:
        print("points that reduces, stuck in local minima")
