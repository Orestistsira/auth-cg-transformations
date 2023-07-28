import numpy as np

from flats import flats
from gourauds import gourauds


def render(verts2d, faces, vcolors, depth, shade_t):
    """
    Sorts triangles by their depth and passes them and draws them on the canvas from the furthest to the
    closest to the camera.

    param verts2d: Holds the coordinates of each vertex (Lx2).
    param faces: Holds the indexes from verts2d array of the vertices for each triangle (Kx3).
    param vcolors: Holds the RGB colors of each vertex (Lx3).
    param depth: Holds the distance from the camera for each vertex (Lx1).
    param shade_t: Sets the shading method to "flat" or "gouraud".
    return: The image with the triangles rendered.
    """

    if shade_t not in ["flat", "gouraud"]:
        print('ERROR: Invalid shading method "' + str(shade_t) + '", choose from ["flat", "gouraud"]')
        exit(1)

    # init image with alla white pixels
    M = 512
    N = 512
    img = [[[1 for _ in range(3)] for _ in range(N)] for _ in range(M)]

    # number of triangles
    K = len(faces)

    # sort triangle faces from farthest from the camera to closest
    triangle_depth = [0.0 for _ in range(K)]

    for triangle in range(K):
        triangle_depth[triangle] = sum(depth[faces[triangle]]) / 3

    # sort in descending order
    sorted_indexes = np.argsort(triangle_depth)[::-1]

    # faces sorted
    faces = faces[sorted_indexes]

    for triangle in range(K):
        tr_indices = faces[triangle]
        tr_vertices = verts2d[tr_indices]
        tr_vcolors = vcolors[tr_indices]
        img = shade_triangle(img, tr_vertices, tr_vcolors, shade_t)

    return img


def shade_triangle(canvas, vertices, vcolors, shade_t):
    """
    Calls the corresponding shading function based on shade_t parameter for the input triangle.

    param canvas: Holds the image data (MxNx3).
    param vertices: Holds the coordinates of the triangle's vertices (3x2).
    param vcolors: Holds the RGB colors of each vertex of the triangle (3x3).
    return: The updated canvas with the triangle shaded.
    return: The image with the input triangle rendered.
    """

    if shade_t == "flat":
        img = flats(canvas, vertices, vcolors)
    else:
        img = gourauds(canvas, vertices, vcolors)

    return img
