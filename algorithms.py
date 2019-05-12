import numpy as np

# Code for Augmented Lagrangian Method
def ALM(A,y,u,alpha,lamda,iters):
    m,n = np.shape(A)
    # initialise x and v of Augmented Lagrangian
    x = np.zeros(n)
    v = np.zeros(m)
    
    # do iterative updates
    err = []
    for k in range(iters):
        xprev = x
        descent = x - alpha*u*np.dot(A.T,np.dot(A,x)-y) - alpha*np.dot(v,A)
        x = np.sign(descent)*np.maximum(np.abs(descent)-lamda,0)     # prox step
        v = v + u*(np.dot(A,x)-y)
        err.append(np.linalg.norm(np.dot(A,x)-y))
        if(np.linalg.norm(x-xprev)<1e-3):
            break

    err_np = np.zeros((iters,))+err[-1]
    err_np[:len(err)] = np.array(err)
        
    return x,v,err_np

# Code for accelerated proximal gradient
def acc_prox_grad(A,y,alpha,beta,lamda,iters):
    m,n = np.shape(A)
    xprev = np.zeros(n)
    x = np.ones(n)
    p = x
    t = 1

    err = []
    for k in range(iters):
        xprev = x
        descent = p - alpha*np.dot(A.T,np.dot(A,p)-y)
        x = np.sign(descent)*np.maximum(np.abs(descent)-lamda*alpha,0)
        beta = (t-1)*2/(np.sqrt(1+4*t**2)+1)
        t = (np.sqrt(1+4*t**2)+1)/2
        p = x + beta*(x - xprev)
        err.append(np.linalg.norm(np.dot(A,x)-y))
        if(np.linalg.norm(x-xprev)<1e-3):
            break
        
    err_np = np.zeros((iters,))+err[-1]
    err_np[:len(err)] = np.array(err)
        
    return x,err_np

def proj_subgrad(A,y,t,iters):
    m,n = np.shape(A)
    x = np.zeros(n)
    
    err = []
    for k in range(iters):
        xprev = x
        z = x - t*np.sign(x)
        x = z - np.dot( np.dot(A.T, np.linalg.inv(np.dot(A,A.T)) ), np.dot(A,z)-y )
        err.append(np.linalg.norm(np.dot(A,x)-y))
        if(np.linalg.norm(x-xprev)<1e-3):
            break

    err_np = np.zeros((iters,))+err[-1]
    err_np[:len(err)] = np.array(err)
        
    return x,err_np

def FrankWolfe(A,y,tau,iters):
    m,n = np.shape(A)
    x = np.zeros(n)
    v = np.zeros(n)
    
    err = []
    for k in range(iters):
        xprev = x
        gamma = 2/(k+2)
        grad = np.dot(A.T,np.dot(A,x)-y)
        #l_inf norm (to find one of the extreme points of L1 ball)
        v = -tau*np.sign(grad)*np.array(np.abs(grad)==np.max(np.abs(grad))) 
        x = (1-gamma)*x + gamma*v
        err.append(np.linalg.norm(np.dot(A,x)-y))
        if(np.linalg.norm(x-xprev)<1e-3):
            break
        
    err_np = np.zeros((iters,))+err[-1]
    err_np[:len(err)] = np.array(err)
        
    return x,v,err_np