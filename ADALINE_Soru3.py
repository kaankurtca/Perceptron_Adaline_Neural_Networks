import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand,randint
from numpy import cos,pi
from adaline_class import Adaline

x1= rand(40,1)
x2=(pi/2)*rand(40,1)
y= np.array([3*x1[i]+2*cos(x2[i]) for i in range(len(x1))])
x=np.concatenate([x1,x2],axis=1)
tumVeriler=np.concatenate([x,y],axis=1)

fig = plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.scatter3D(tumVeriler[:,0],tumVeriler[:,1],tumVeriler[:,2],'green')
plt.show()

egitimBoyutu=round(tumVeriler.shape[0]*0.8)
randomİndisler = np.random.choice(tumVeriler.shape[0], size=egitimBoyutu, replace=False)
egitimVerisi = tumVeriler[randomİndisler, :]
testVerisi = np.delete(tumVeriler, randomİndisler, axis=0)

x_egitim = egitimVerisi[:, :-1]
y_egitim = (egitimVerisi[:, -1]).reshape(egitimBoyutu, 1)
x_test = testVerisi[:, :-1]
y_test = (testVerisi[:, -1]).reshape(tumVeriler.shape[0]-egitimBoyutu, 1)

ada=Adaline()

agirlik=ada.egit(x_egitim, y_egitim,5)
tahmin = ada.tahminEt(x_egitim)
r2 = ada.ortKareHata(tahmin,y_egitim)
print(r2)