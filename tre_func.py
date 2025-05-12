import numpy as np
def distanceFromLine(p1,p2,p3):
    d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
    return d

def unit(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)