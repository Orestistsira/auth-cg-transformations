import numpy as np

from render import render


def ChangeCoordinateSystem(cp, R, c0):
    """
    Transforms a point or an array of points cp in R^3 (in non-homogeneous form) to a new coordinate system dp using a
    rotation matrix R and a translation vector c0.

    param cp: Point or array of points in R^3 (in non-homogeneous form) with respect to the original coordinate system.
    param R: Rotation matrix from the original coordinate system to the new coordinate system.
    param c0: Translation vector representing the origin of the new coordinate system with respect to the original.
    return dp: Transformed point or array of points in R^3 (in non-homogeneous form) with respect to the new coordinate
    system.
    """

    # Compute the new point(s) in the new coordinate system by applying the rotation and translation transformations
    dp = R @ (cp - c0)

    return dp


def PinHole(f, cv, cx, cy, cz, p3d):
    """
    Computes the perspective view of a point or an array of points p3d in R^3 (in nonhomogeneous form) using a pinhole camera model.

    param f: Distance of the curtain from the center (measured in the units used by the camera coordinate system).
    param cv: Coordinates of the center of the perspective camera with respect to the WCS.
    param cx: Unit vector x of the perspective camera with respect to the WCS.
    param cy: Unit vector y of the perspective camera with respect to the WCS.
    param cz: Unit vector z of the perspective camera with respect to the WCS.
    param p3d: Point or array of points in R^3 (in nonhomogeneous form) with respect to the WCS.
    return p2d: Perspective view of the point or array of points p3d in 2D.
    return depth: Depth of each point in p3d in the perspective camera view.
    """

    # Define the camera coordinate system
    R = np.array([cx, cy, cz])
    c0 = cv.reshape(3, 1)

    # Transform the points from WCS to the camera coordinate system
    p_cam = ChangeCoordinateSystem(p3d, R, c0)

    # Compute depth of each point
    depth = p_cam[2, :]

    # Compute perspective projection
    # p2d = (f / depth) * p_cam[:2, :]
    p2d = f * (p_cam[:2, :] / depth)

    return p2d, depth


def CameraLookingAt(f, cv, cK, cup, p3d):
    """
    Computes the perspective views and depth of the 3D points p3d using a pinhole camera model,
    where the camera is looking at a target point K and the up vector is given.

    param f: Distance of the curtain from the center (measured in the units used by the camera coordinate system).
    param cv: Coordinates of the center of the camera with respect to the WCS.
    param cK: Coordinates of the target point K with respect to the WCS.
    param cup: Unit vector representing the up direction of the camera.
    param p3d: Point or array of points in R^3 (in nonhomogeneous form) with respect to the WCS.
    return p2d: Perspective view of the point or array of points p3d in 2D.
    return depth: Depth of each point in p3d in the perspective camera view.
    """

    # Compute the camera coordinate system
    zc = (cK - cv) / np.linalg.norm(cK - cv)
    t = cup - (cup @ zc) * zc
    yc = t / np.linalg.norm(t)
    xc = np.cross(yc, zc)

    # Call PinHole function to project 3D points
    p2d, depth = PinHole(f, cv, xc, yc, zc, p3d)

    # print('p2d:')
    # print(p2d)
    # print('depth:')
    # print(depth)

    return p2d, depth


def rasterize(p2d, Rows, Columns, H, W):
    """
    Converts the 2D camera coordinate points to their corresponding pixel positions in the image.

    param p2d: 2D camera coordinate points.
    param Rows: Number of rows in the image.
    param Columns: Number of columns in the image.
    param H: Height of the camera's curtain in inches.
    param W: Width of the camera's curtain in inches.
    return n2d: Pixel positions of the points in the image.
    """

    # Scale the camera coordinate points to match pixel positions
    scaled_x = ((p2d[0, :] + W / 2) / W) * Columns
    # scaled_y = ((H / 2 - p2d[1, :]) / H) * Rows
    scaled_y = ((p2d[1, :] + H / 2) / H) * Rows

    # Round the scaled coordinates to the nearest integers
    n2d = np.round(np.vstack((scaled_x, scaled_y))).astype(int)

    return n2d


def RenderObject(p3d, faces, vcolors, H, W, Rows, Columns, f, cv, cK, cup):
    """
    Renders an object from the cameras' perspective using gouraud's method

    param p3d: Point or array of points in R^3 (in nonhomogeneous form) with respect to the WCS.
    param faces: Holds the indexes from p3d array of the vertices for each triangle (Kx3).
    param vcolors: Holds the RGB colors of each vertex (Lx3).
    param H: Height of the camera's curtain in inches.
    param W: Width of the camera's curtain in inches.
    param Rows: Number of rows in the image.
    param Columns: Number of columns in the image.
    param f: Distance of the curtain from the center (measured in the units used by the camera coordinate system).
    param cv: Coordinates of the center of the camera with respect to the WCS.
    param cK: Coordinates of the target point K with respect to the WCS.
    param cup: Unit vector representing the up direction of the camera.
    return I: The image of the rendered object from the camera view.
    """

    # Project 3D points to 2D camera coordinate system
    p2d, depth = CameraLookingAt(f, cv, cK, cup, p3d)

    # Convert the 2D camera coordinate points to their corresponding pixel in the camera.
    p2d = rasterize(p2d, Rows, Columns, H, W)

    # Render the points from the camera's perspective
    I = render(p2d.T, faces, vcolors, depth, "gouraud")

    return I
