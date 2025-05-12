import cvxpy as cp
import numpy as np
np.set_printoptions(suppress=True)
import random as ran



import numpy as np
from scipy.optimize import minimize
from functions import distanceFromLine,unit
from numpy.linalg import norm
from scipy import optimize





def compute_tre(vec):
    vector = vec.reshape(4,3)
    target=np.array([0,160,0])

    # add noise to system
    # vector = vector + noise

    centroid = np.mean(vector,axis=0)

    axis =[]
    temp_vec =  unit(vector[1] -  centroid)
    temp_vec2 = np.cross(temp_vec,unit(vector[2]-centroid))
    temp_vec3 = np.cross(temp_vec,temp_vec2)
    axis.append(temp_vec)
    axis.append(temp_vec2)
    axis.append(temp_vec3)


    # fiducials to principal axis
    # target from principal axis 
    tp = []
    tp.append(distanceFromLine(centroid,centroid + axis[0],target))
    tp.append(distanceFromLine(centroid,centroid + axis[1],target))
    tp.append(distanceFromLine(centroid,centroid + axis[2],target))

    fp = []
    f = np.zeros(3)
    for i in range(3):
        sum = 0
        for j in range(len(vector)):
            fp = np.around(distanceFromLine(centroid,centroid+axis[i],vector[j]),3)
            # print(fp)
            sum = sum + fp**2
        if np.round(sum/len(vector)) == 0:
            f[i]=10000
            # print(f[i])
        else:
            f[i] = tp[i]**2/(sum/len(vector))
            
        
        

    fle = 0.3
    tre = (fle**2/len(vector))*(np.mean(f)+1)
    # if np.sum(norm(vector,axis=1)>threshold):
    #     return 100
    # else:
    return np.sqrt(np.around(tre,3))


#Generate
height = 345
width = 325
samples = 1000
x = width/2
y = height/2
z = 0 
q1_1  = np.array([0,x,z])
q1_2  = np.array([x,y,z])
q1_3  = np.array([x,0,z])

q2_1  = np.array([0,-x,z])
q2_2  = np.array([x,-y,z])
q2_3  = np.array([x,0,z])

q3_1  = np.array([0,-x,z])
q3_2  = np.array([-x,-y,z])
q3_3  = np.array([-x,0,z])

q4_1  = np.array([0,y,z])
q4_2  = np.array([-x,y,z])
q4_3  = np.array([-x,0,z])

B1=np.array([q1_1,q1_2,q1_3])
B2=np.array([q2_1,q2_2,q2_3])
B3=np.array([q3_1,q3_2,q3_3])
B4=np.array([q4_1,q4_2,q4_3])
hulls = [B1,B2,B3,B4]

def generate_B(B,r):
    B = np.array(B)
    df = []
    vdata = []

    l = []
    # r = [ran.random() for i in range(len(B))]
    r = np.array([r])
    r = r/r.sum()
    x_ = B.transpose()@r.transpose()
    # x_.transpose()[0]
    return x_.transpose()[0]

print(generate_B(hulls[0],[1,2,3]))
# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
# objective = cp.Minimize(cp.sum_squares(A @ x - b))
vec = cp.Variable(4,3)
a1 = generate_B(hulls[0],np.random.randint(5, size=(4, 3)))

fids = np.hstack(generate_B(hulls[0],[1,2,5]))
objective = cp.Minimize(compute_tre())

constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)



# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)