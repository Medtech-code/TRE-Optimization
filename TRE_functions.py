import plotly.graph_objects as go
import numpy as np

def compute_ortho(D):
    X=D[0]
    Y=np.cross(X,D[1])
    Z=np.cross(X,Y)
    vector=np.vstack((X,Y,Z))
    X,Y,Z=vector[0],vector[1],vector[2]
    i=X/(X**2).sum()**0.5
    j=Y/(Y**2).sum()**0.5
    k=Z/(Z**2).sum()**0.5
    # print("unit vector=",i,j,k)
    return np.array([i,j,k])


def rms(v):
    temp = np.array(v)
    return (temp**2).sum()**0.5

def compute_targetdist(point,line_pt1,line_pt2): 
    point_vec = point - line_pt1
    line_vec = line_pt2-line_pt1 
    return np.linalg.norm(np.cross(point_vec,line_vec)/np.linalg.norm(line_vec))




def plot_fids(fids):
    # fids=np.vstack((fids,[0, 0, 0]))
    fiducials = go.Scatter3d(
        x=fids[:,0], y=fids[:,1], z=fids[:,2],
        marker=dict(size=4,colorscale='Viridis',),line=dict(color='darkblue',width=2))
    axes = go.Scatter3d(x = [0,0,0,100,0,0],y = [0, 100,0,0,0,0 ],z=[0,0,0,0,0,100],marker = dict( size = 1,color = "rgb(84,48,5)"),line=dict(color="rgb(84,48,5)",width=6))
    data = [fiducials,axes]
    name = 'default'
# Default parameters which are used when `layout.scene.camera` is not provided
    camera = dict(up=dict(x=-1, y=0, z=0),center=dict(x=0, y=0, z=0),eye=dict(x=0, y=0, z=1.25))

    fig = go.Figure(data=data)
    fig.update_layout(scene_camera=camera, title=name)
    fig.update_layout(scene = dict(xaxis = dict(nticks=4, range=[-100,100],),yaxis = dict(nticks=4, range=[-100,100],),zaxis = dict(nticks=4, range=[-100,100],),),
    width=700,
    margin=dict(r=20, l=10, b=10, t=10))
    return fig