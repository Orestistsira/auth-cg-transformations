import numpy as np


def rotmat(theta, u):
    """
    Computes the rotation matrix R corresponding to a clockwise rotation through an angle theta in radians
    about an axis with a direction given by the unit vector u.

    param theta: Angle of rotation in radians.
    param u: 1x3 Unit vector representing the axis of rotation.
    return: 3x3 rotation matrix corresponding to the given rotation.
    """

    # Compute the components of the rotation matrix using the Rodrigues' rotation formula
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    x, y, z = u

    R = np.array([[t*x**2+c,  t*x*y-s*z, t*x*z+s*y],
                  [t*x*y+s*z, t*y**2+c,  t*y*z-s*x],
                  [t*x*z-s*y, t*y*z+s*x, t*z**2+c]])

    return R


def RotateTranslate(cp, theta, u, A, t):
    """
    Transforms a point or an array of points cp in R^3 (in non-homogeneous form), by rotating it by an angle theta in
    radians about an axis passing through the point and parallel to u, and then shifting it by a displacement vector t.
    The rotation is performed about the WCS.

    param cp: Point or an array of points in R^3 (in non-homogeneous form) (3xN).
    param theta: Angle of rotation in radians.
    param u: Unit vector representing the direction of the axis of rotation (3x1).
    param A: Point about which the rotation is performed.
    param t: Displacement vector.
    return cq: Transformed point or array of points in R^3 (in non-homogeneous form).
    """

    # Compute the rotation matrix
    R = rotmat(theta, u.T[0])

    # Translate the point(s) to the origin
    cp = cp - A
    cq = R @ cp

    # Translate the point(s) back to their original position and apply the displacement vector
    cq = cq + A + t

    return cq
