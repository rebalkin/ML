import numpy as np

def z_boost(beta):

    if np.abs(beta)>1:
        raise ValueError(f"beta must satisfy |beta| ≤ 1, got {beta}")
    boost = np.eye(4)
    eta = np.atanh(beta)
    boost[0,0] = np.cosh(eta)
    boost[3,3] = np.cosh(eta)
    boost[0,3] = np.sinh(eta)
    boost[3,0] = np.sinh(eta)
    

    return boost

def rotate(theta,phi):

    R1 = np.eye(4)
    R2 = np.eye(4)

    R1[1,1] = np.cos(theta)
    R1[3,3] = np.cos(theta)
    R1[1,3] = np.sin(theta)
    R1[3,1] = -np.sin(theta)

    R2[1,1] = np.cos(phi)
    R2[2,2] = np.cos(phi)
    R2[1,2] = -np.sin(phi)
    R2[2,1] = np.sin(phi)

    return np.dot(R2,R1)

def random_boost():

    beta = np.random.uniform(-1, 1)
    costheta = np.random.uniform(-1, 1)
    phi = np.random.uniform(0, 2*np.pi)

    boost = z_boost(beta)
    R = rotate(np.acos(costheta),phi)

    return np.dot(R,boost)

def generate_4_mom(mass, N):

    if mass>0:
        rest = np.array([mass,0,0,0])
    else:
        rest = np.array([1,0,0,1])
    
    boosts = np.zeros([N,4,4])
    for i in range(N):
        boosts[i] = random_boost()

    return np.einsum('nij,j->ni', boosts, rest)

def generate_training_set(m1, m2, N1, N2):

    vec1 = generate_4_mom(m1, N1)
    vec2 = generate_4_mom(m2, N2)

    X = np.concatenate([vec1, vec2], axis=0)

    res1 = np.ones(N1)  
    res2 = np.zeros(N2)

    Y  = np.concatenate([res1, res2], axis=0)

    return X, Y

