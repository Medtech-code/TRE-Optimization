import numpy as np
from random import sample
import matplotlib.pyplot as plt
import math
from functions_1 import distanceFromLine
from sympy import *
import sympy
from sympy.vector import CoordSys3D,matrix_to_vector,Vector
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import Mul
a = 160
b = 80
c= 0
alpha = 5
search_sample = 12
desired_tre = 2
cound_fid = 3
noOfSteps = 100
target=Matrix(1,3,[-100,30,40])
# def create_mesh(a,b):
xx = np.linspace(0, a, 9)
yy = np.linspace(-b/2, b/2, 9)
theta, phi = np.meshgrid(xx, yy)
u,v,w = symbols('u v w')
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
fiducials= Matrix(mesh[l,:])
expr_var = Matrix()
C=CoordSys3D('C')


target_pos=Mul(target)
height  = np.array([0,a])
width = np.array([-b/2,b/2])
origin = Matrix(3,1,[0,0,0])
axis = eye(3)
# j is len of axis
for j in range(3):
    val = matrix_to_vector(target_pos,C).cross(matrix_to_vector(axis[:,j],C))
#    target distance contains distance of the target from the i,j,k axis
    
 
   
    fid_dist = Vector.zero
    for i in range(fiducials.shape[0]):
      fid_dist = fid_dist+ (matrix_to_vector(fiducials[i,:],C).cross(matrix_to_vector(axis[:,j],C)))

    if j == 0:
    #   sum of target/3 fiducial distance from axis[0]
      sum_fid_x =sympy.div(val,fid_dist)
    elif j==1:
    #   sum of target/3 fiducial distance from axis[1]
       sum_fid_y = sympy.div(val,fid_dist)
    elif j==2:
    #   sum of target/3 fiducial distance from axis[2]
       sum_fid_z = sympy.div(val,fid_dist)
      
fle = 0.3
N = 3  
fle**2(1+1/N())
# component_1=sum_of_target[0]/sum_fid_x
# component_2=sum_of_target[1]/sum_fid_y
# component_3=sum_of_target[2]/sum_fid_z
    
# print("component1=",component_1)
# print("component2=",component_2)
# print("component3=",component_3)    
# ##taking partial derivative for component 1 using sympy
# # take partial derivative with respect to v
# derivative_component_1_v=simplify(component_1.diff(v))
# # take partial derivative with respect to 
# derivative_component_1_w=simplify(component_1.diff(w)) 
# derivative_component_2_u=simplify(component_2.diff(u))
# derivative_component_2_w=simplify(component_2.diff(w))  
# derivative_component_3_u=simplify(component_3.diff(u))       
# derivative_component_3_v=simplify(component_3.diff(v)) 
# print(derivative_component_3_u)
components=[
            sum_of_target[0]/sum_fid_x,
            sum_of_target[1]/sum_fid_y,
            sum_of_target[2]/sum_fid_z,
]
variables=[u,v,w] 
derivatives=[] 
for component in components:
    component_derivatives=[]

    for variable in variables:
        component_derivative=simplify(component.diff(variable))
        component_derivatives.append(component_derivative)
    derivatives.append(component_derivatives) 

u_value=axis[0]
v_value=axis[1]
w_value=axis[2]
gradient=[]

for i,component_derivatives in enumerate(derivatives):
    for j,variable in enumerate(variables):
        print(f"Derivative of component {i + 1} with respect to {variable}: {component_derivatives[j]}")

    sum_of_derivatives=simplify(Add(component_derivatives[0],component_derivatives[1],component_derivatives[2]))
    gradient.append(sum_of_derivatives)









            

           
  



            



