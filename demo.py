import numpy as np
import matplotlib.pyplot as plt

from render_object import RenderObject
from rotate_translate import RotateTranslate


# Load data from file
data = np.load('h2.npy', allow_pickle=True)[()]
verts3d = data['verts3d']
vcolors = data['vcolors']
faces = data['faces']
c_org = data['c_org'].T[0]
c_lookat = data['c_lookat'].T[0]
c_up = data['c_up'].T[0]
f = data['focal']
t_1 = data['t_1']
t_2 = data['t_2']
u = data['u']
phi = data['phi']

print(vcolors)

# Set image size
Rows = 512
Columns = 512

# Set camera's curtain size
H = 15
W = 15

# Set t_1 and t_2
t_1 = np.array([t_1]).T
t_2 = np.array([t_2]).T

# Render initial object
print('Rendering initial image [0.jpg]...')
I = RenderObject(verts3d, faces, vcolors, H, W, Rows, Columns, f, c_org, c_lookat, c_up)
plt.imsave('0.jpg', np.array(I[::-1]))

# Move object by t_1 and render it
print('Rendering 1st transformation image [1.jpg]...')
verts3d = verts3d + t_1
I = RenderObject(verts3d, faces, vcolors, H, W, Rows, Columns, f, c_org, c_lookat, c_up)
plt.imsave('1.jpg', np.array(I[::-1]))

# Rotate object by phi rads about an axis parallel to u and render it
print('Rendering 2nd transformation image [2.jpg]...')
u = np.array([u]).T
verts3d = RotateTranslate(verts3d, phi, u, np.array([[0, 0, 0]]).T, np.array([[0, 0, 0]]).T)
I = RenderObject(verts3d, faces, vcolors, H, W, Rows, Columns, f, c_org, c_lookat, c_up)
plt.imsave('2.jpg', np.array(I[::-1]))

# Move object by t_2 and render it
print('Rendering 3rd transformation image [3.jpg]...')
verts3d = verts3d + t_2
I = RenderObject(verts3d, faces, vcolors, H, W, Rows, Columns, f, c_org, c_lookat, c_up)
plt.imsave('3.jpg', np.array(I[::-1]))
