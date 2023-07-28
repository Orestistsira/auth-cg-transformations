import numpy as np

from interpolate_vectors import interpolate_vectors
from update_active_points import update_active_points


def gourauds(canvas, vertices, vcolors):
    """
    Renders the passed triangle on the canvas using the "gouraud" method.

    param canvas: Holds the image data (MxNx3).
    param vertices: Holds the coordinates of the triangle's vertices (3x2).
    param vcolors: Holds the RGB colors of each vertex of the triangle (3x3).
    return: The updated canvas with the triangle shaded.
    """

    ykmin = [0 for _ in range(3)]
    ykmax = [0 for _ in range(3)]

    # we find min, max for each edge k = [0, 2]
    for k in range(3):
        ykmin[k] = min(vertices[k][1], vertices[(k + 1) % 3][1])
        ykmax[k] = max(vertices[k][1], vertices[(k + 1) % 3][1])

    ymin = min(ykmin)
    ymax = max(ykmax)

    active_points = []
    mk = [0.0 for _ in range(3)]

    # we find the list of active points
    for k in range(3):
        start_point = vertices[k]
        end_point = vertices[(k + 1) % 3]

        # grad
        if (end_point[0] - start_point[0]) == 0.0:
            mk[k] = np.inf
        else:
            mk[k] = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])

        if start_point[1] == ymin:
            if mk[k] != 0:
                active_points.append([start_point[0], start_point[1], k])
        if end_point[1] == ymin:
            if mk[k] != 0:
                active_points.append([end_point[0], end_point[1], k])

    # loop for each y horizontal line between ymin and ymax
    for y in range(ymin, ymax):
        # sort active points by x
        active_points = sorted(active_points, key=lambda l: l[0])

        # calculate A, B and their colors for line y
        A, B, A_color, B_color = calculate_colors(active_points, vertices, vcolors)

        for x in range(round(active_points[0][0]), round(active_points[-1][0])):
            color = interpolate_vectors([A[0], A[1]], [B[0], B[1]], A_color, B_color, x, 1)
            color = np.clip(color, 0, 1)
            canvas[y][x] = color

        # update active points
        active_points = update_active_points(y, ykmin, ykmax, active_points, vertices, mk)

    return canvas


def calculate_colors(active_points, vertices, vcolors):
    """
    Calculates the coordinates and the color of the two points that the scanning line crosses the triangle, using the
    vector interpolation function.

    param active_points: Array that holds the coordinates and the edge of each current active point.
    param vertices: Holds the coordinates of the triangle's vertices (3x2).
    param vcolors: Holds the RGB colors of each vertex (Lx3).
    return: The coordinates and the color of each point.
    """

    # get the first and the last point of active_points array
    A = active_points[0]
    B = active_points[-1]

    A_start = A[2]
    A_end = (A_start + 1) % 3
    A_color = interpolate_vectors(vertices[A_start], vertices[A_end], vcolors[A_start], vcolors[A_end], A[1], 2)

    B_start = B[2]
    B_end = (B_start + 1) % 3
    B_color = interpolate_vectors(vertices[B_start], vertices[B_end], vcolors[B_start], vcolors[B_end], B[1], 2)

    return A, B, A_color, B_color
