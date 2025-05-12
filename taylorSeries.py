import numpy as np
import matplotlib.pyplot as plt
# approximation of a function f(x)=x**3
# define the function
def x_cube(a):
    return a**3

# define the derivative of the function
def x_cube_nth_derivative_at_a(n,a):
    if n==1:
        return 3*a**2
    
    if n==2:
        return 6*a
    
    if n==3:
        return 6
    
    if n>3:
        return 0
    
# define the factorial function
def factorial(n):
    if n<1:
        return 1
    
    else:
        return n*factorial(n-1)
    
# Function approximating x**3 with m terms of taylor series
def xcube_approx(x,m):
    a=1
    func_value=x_cube(a)
    for n in range (1,m+1):
        func_value += (x_cube_nth_derivative_at_a(n,a)/factorial(n))*pow(x-a,n)
    return func_value


# evaluate how close thew approximation is
# generate x values
# numpy.arange([start, ]stop, [step, ])
x_vec=np.arange(-5,5,0.1)

y_vec_approx_1=[xcube_approx(x,1) for x in x_vec]
y_vec_approx_2=[xcube_approx(x,2) for x in x_vec]
y_vec_approx_3=[xcube_approx(x,3) for x in x_vec]

# groundtruth value of f(x)
y_correct=[x**3 for x in x_vec]
# runtime configuration setting in matplotlib 
# set the fig size
plt.rcParams['figure.figsize']=15,4
# The plot has 1 row 4 columns and it is the first plot
plt.subplot(1,4,1)
plt.plot(x_vec,y_correct)
plt.ylabel("$f(x)")
plt.xlabel('$x')
plt.title('groundtruth plot for the function')


# Second subplot-approximation with one term
plt.subplot(1,4,2)
plt.plot(x_vec,y_vec_approx_1)
plt.xlabel('$x')
plt.ylabel('$f(x)')
plt.title("$x^3$ Approx. with $m$=1")


# Third plot-approximation with 2 terms
plt.subplot(1,4,3)
plt.plot(x_vec,y_vec_approx_2)
plt.xlabel('$x')
plt.ylabel('$f(x)')
plt.title("$x^3$ Approx. with $m$=1")

# Fourth Plot-approximation with 3 terms
plt.subplot(1,4,4)
plt.plot(x_vec,y_vec_approx_3)
plt.xlabel('$x')
plt.ylabel('$f(x)')
plt.title("$x^3$ Approx. with $m$=1")
plt.show()


# Newtons method which is an extension of taylors methor as finding a root of the function works iteratively starting from a initial value x0 by using the first order expansion of f around xi

def find_root(f,f_deri,x0,max_steps=10,delta=0.001,verbose=True):
    x_old=x0
    for i in range(max_steps):
        try:
            x_new=x_old-f(x_old)/f_deri(x_old)

        except ZeroDivisionError:
            return None
        

        if verbose:print("Iteration",i,"(x_old,x_new):",(x_old-x_new),"|x_new-x_old|:",abs(x_new-x_old))
        if abs(x_new-x_old)< delta:break

        x_old=x_new

    return x_new
    

# try this with a function

def f(x):
    return x**2-4

def f_derv(x):
    return 2*x
# # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source]
def draw_f():
    x=np.linspace(-3,3,50)
    # specify the width and height of the figure with tuple bracket
    plt.rcParams['figure.figsize']=5,4
    plt.plot(x,f(x))
    plt.title("$f(x)=x^2-4$")
    


# test the newtons method for finding the roots
draw_f()
for i in range(20):
    x0=np.random.randint(-5,5)
    r=find_root(f,f_derv,x0,max_steps=50,delta=0.001,verbose=False)
    if (not r) or (abs(f(r)) > 0.001): continue
    print("trial",i,"ended with root",r,"value of the root",f(r))
    plt.scatter(r,f(r))
plt.show()


# Newtons  method in scipy
# syntax scipy.optimize.newton


# Newtons method for finding the local minima
def find_local_minima(f_first_deriv,f_second_deriv,x0,max_steps=10,delta=0.001,Verbose=True):
    x_old=x0
    for i in range(max_steps):
        try:
            x_new= x_old-(f_first_deriv(x_old)/f_second_deriv(x_old))

        except ZeroDivisionError:
         return None
        

        if Verbose:print("Iteration",i,"(x_old,x_new)",(x_old,x_new),"|x_new-x_old|",abs(x_new-x_old))
        if abs(x_new-x_old)<delta:break

        x_old=x_new
    return x_new

def f(x):
    return x**2-4

def f_first_deriv(x):
    return 2*x

def f_second_deriv(x):
    return 2

draw_f()
for i in range(10):
    x0=np.random.randint(-5,5)
    x_min=find_local_minima(f_first_deriv,f_second_deriv,x0,max_steps=10,delta=0.0001,Verbose=True)
    if (x_min is None) or (abs(f_first_deriv(x_min)) > 0.0001):continue
    print("trial",i,"x_min",x_min,"f'(x_min)",f_first_deriv(x_min))
    plt.scatter(x_min,f(x_min))
plt.show()

# scipy for findng minima using Newtons method
# scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(), method=None, tol=None, options=None)[source]
# fun=the objective function,bounds=For method ‘bounded’, bounds is mandatory and must have two finite items corresponding to the optimization bounds,method=Brent,bounded,golden,tol=delta value
