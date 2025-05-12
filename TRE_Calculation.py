import numpy as np
import matplotlib.pyplot as plt
from TRE_functions import compute_axis,compute_unit,rms,compute_targetdist,plot_fids
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
from TRE_functions import compute_axis,compute_unit,rms,compute_targetdist,plot_fids

fiducial=np.array([[1,2,0],[1,-1,5],[-1,1,0]])   
target=np.array([[-1,-1,0]])
def compute_TRE(fiducial,target):
    xc=np.sum(fiducial[:3],axis=0)/3
    D=np.subtract(fiducial,xc)
    compute_axis(D)
    compute_unit(D)
    axis=compute_unit(D)
    target_dist = []
    for i in range(3):
        origin = xc
        line_pt1 = origin
        line_pt2 = origin + axis[i]
        point = target
        compute_targetdist(point,line_pt1,line_pt2)
        target_dist.append(compute_targetdist(point,line_pt1,line_pt2))
    print(f'target distances :{target_dist}')
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
    print(datax,datay,dataz)
    d = target_dist
    f = np.zeros((3,1))
    f[0]=rms(datax)
    f[1]=rms(datay)
    f[2]=rms(dataz)
    print(f)
    N =len(fiducial)
    gamma = np.square(d[0]/rms(datax))+np.square(d[1]/rms(datay))+np.square(d[2]/rms(dataz))
    tre = np.sqrt((1+gamma/3)/N)
    print("tre=",tre)
    return tre

tre=compute_TRE(fiducial,target)