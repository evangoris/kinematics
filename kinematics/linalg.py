"""Utility routines for linear algebra in homogeneous 2D space

"""
import numpy as np
import numpy.linalg as LA

def cross(u, v):
    return u[0]*v[1] - u[1]*v[0]

def inner(u, v):
    return u[0]*v[0] + u[1]*v[1]

def angle(u, v):
    from math import copysign, acos
    return copysign(acos(inner(u, v)/(LA.norm(u)*LA.norm(v))), cross(u, v))

def transrot_matrix(alpha, x, y, out=None):
    """Construct a matrix that performs a
    translation followed by a rotation
    """
    from math import cos, sin
    cosa = cos(alpha)
    sina = sin(alpha)

    A = np.empty((3, 3)) if out is None else out
    A[0, :] = [cosa, -sina, cosa*x - sina*y]
    A[1, :] = [sina, cosa, sina*x + cosa*y]
    A[2, :] = [0, 0, 1]

    return A

def rottrans_matrix(alpha, x, y, out=None):
    """Construc a matrix that performs a rotation
    followed by a translation
    """
    from math import cos, sin
    cosa = cos(alpha)
    sina = sin(alpha)

    A = np.empty((3, 3)) if out is None else out
    A[:, :] = [[cosa, -sina, x], [sina, cosa, y], [0, 0, 1]]

    return A

def rotationmatrix(alpha):
    from math import sin, cos
    return np.array([[cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0], [0, 0, 1]])

def translationmatrix(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

def degree_to_radian(deg):
    import math
    return float(deg)*math.pi/180.0

def radian_to_degree(rad):
    import math
    return float(rad)*180.0/math.pi

def dot(A, v):
    x, y, _ = np.dot(A, v + (1,))
    return (x, y)

def rottrans_inv(m):
    R = np.array([[m[0,0], m[1,0], 0], [m[0,1], m[1,1], 0], [0, 0, 1]])
    T = np.array([[1, 0, -m[0,2]], [0, 1, -m[1,2]], [0, 0, 1]])
    return np.dot(R, T)


