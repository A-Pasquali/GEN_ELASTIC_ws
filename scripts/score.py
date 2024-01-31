from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

A = np.array([0,0])
B = np.array([600,600])
Kpoint = np.array([600,0])
K = 10

def distance(p1, p2):
    return np.linalg.norm(p1-p2)*0.001

def force(p):
    return K * distance(p, Kpoint)

def f(x, y):
    distanceAB = distance(A, B)
    good_distance = distanceAB/2
    print(good_distance)
    score_dist = 0.5*abs(np.sqrt((x-A[0])**2 + (y-A[1])**2)*0.001 - good_distance) + 0.5*abs(np.sqrt((x-B[0])**2 + (y-B[1])**2)*0.001 - good_distance)
    score_force = abs(np.sqrt((x-Kpoint[0])**2 + (y-Kpoint[1])**2)*0.001)*K
    #return (1/(distanceAB - score_dist)) * (1/(30-score_force))
    Sd = 1+score_dist
    Sf = 1+(score_force)/50
    return 0.25*Sf*(Sd)**2

x = np.linspace(0, 600, 300)
y = np.linspace(0, 600, 300)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

#print x,y that gives the lowest score 0-600
xmin, ymin = np.unravel_index(np.argmin(Z), Z.shape)
x_min_value = x[xmin]
y_min_value = y[ymin]
z_min_value = Z[xmin, ymin]

print("x_min_value: ", x_min_value)
print("y_min_value: ", y_min_value)

ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color=np.array([0.3, 0.3, 0.7, 0.3]))
ax.scatter(y_min_value, x_min_value, z_min_value, color=np.array([0.1, 0.6, 0.6, 0.8]), marker='o', s=100)
#disegan una curva sul piano x,y con z = 0 tra i punti A, il minimo e B
x_curve1 = np.linspace(A[0], y_min_value, 100)
y_curve1 = np.linspace(A[1], x_min_value, 100)
z_curve1 = np.zeros(100)
ax.plot3D(x_curve1, y_curve1, z_curve1, color=np.array([0.1, 0.6, 0.6, 0.8]), linewidth=3)
x_curve2 = np.linspace(y_min_value, B[0], 100)
y_curve2 = np.linspace(x_min_value, B[1], 100)
z_curve2 = np.zeros(100)
ax.plot3D(x_curve2, y_curve2, z_curve2, color=np.array([0.1, 0.6, 0.6, 0.8]), linewidth=3)
ax.scatter(A[0], A[1], 0, color=np.array([0.1, 0.6, 0.6, 0.8]), marker='o', s=100)
ax.scatter(B[0], B[1], 0, color=np.array([0.1, 0.6, 0.6, 0.8]), marker='o', s=100)
ax.scatter(y_min_value, x_min_value, 0, color=np.array([0.1, 0.6, 0.6, 0.8]), marker='o', s=100)
ax.scatter(Kpoint[0], Kpoint[1], 0, c='purple', marker='*', s=200)
#vertical line from y_min_value, x_min_value
x_curve3 = np.linspace(y_min_value, y_min_value, 100)
y_curve3 = np.linspace(x_min_value, x_min_value, 100)
z_curve3 = np.linspace(0, z_min_value, 100)
ax.plot3D(x_curve3, y_curve3, z_curve3, color=np.array([0.1, 0.6, 0.6, 0.8]), linestyle='dashed', linewidth=3)
#line from Kpoint to y_min_value, x_min_value
x_curve4 = np.linspace(Kpoint[0], y_min_value, 100)
y_curve4 = np.linspace(Kpoint[1], x_min_value, 100)
z_curve4 = np.zeros(100)
ax.plot3D(x_curve4, y_curve4, z_curve4, 'purple', linestyle='dashed', linewidth=3)
ax.set_title('surface')
plt.show()