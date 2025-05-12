import numpy as np
from numpy.linalg import norm

def distanceFromLine(p1,p2,p3):
    d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    return d

def unit(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def compute_tre(vector,target):
    vector = np.array(vector)
    centroid = np.mean(vector,axis=0)
    axis =[]
    temp_vec =  unit(vector[1] -  centroid)

    temp_vec2 = unit(np.cross(temp_vec,unit(vector[2]-centroid)))
    temp_vec3 = unit(np.cross(temp_vec,temp_vec2))
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
    return np.sqrt(np.around(tre,3))