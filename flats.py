import numpy as np

from update_active_points import update_active_points


def flats(canvas, vertices, vcolors):
    """
    Renders the passed triangle on the canvas using the "flat" method.

    param canvas: Holds the image data (MxNx3).
    param vertices: Holds the coordinates of the triangle's vertices (3x2).
    param vcolors: Holds the RGB colors of each vertex of the triangle (3x3).
    return: The updated canvas with the triangle shaded.
    """

    r = sum([vcolors[i][0] for i in range(3)]) / 3
    g = sum([vcolors[i][1] for i in range(3)]) / 3
    b = sum([vcolors[i][2] for i in range(3)]) / 3

    color = [r, g, b]

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

    # we find the list of active points and the slopes of the edges
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

        for x in range(round(active_points[0][0]), round(active_points[-1][0])):
            canvas[y][x] = color

        # update active points
        active_points = update_active_points(y, ykmin, ykmax, active_points, vertices, mk)

    return canvas
