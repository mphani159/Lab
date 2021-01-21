import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-5, 4.5, 100)
ylist = np.linspace(-5, 4.5, 200)
X, Y = np.meshgrid(xlist, ylist)
#Z = np.sqrt(X**2 + Y**2)
Z = np.sqrt(X**2 + Y**2)

fig,ax=plt.subplots(1,1)
ax = plt.axes(projection='3d')
ax.view_init(20, 35)
cp = ax.contour3D(X, Y, Z, 50)
#fp = plt.plot(X,Y,Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Contours Plot')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_zlabel('z (cm)')
plt.show()

#ftnp=plt.plot(Z)
#plt.show()