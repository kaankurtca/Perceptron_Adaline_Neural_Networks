import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand,randint
from numpy import cos,pi
from Madaline_Class import Madaline

x1= rand(50,1)
x2=(pi/2)*rand(50,1)
y= np.array([3*x1[i]+2*cos(x2[i]) for i in range(len(x1))])
x=np.concatenate([x1,x2],axis=1)
bias=np.ones((len(x),1))
x_bias=np.concatenate([x,bias],axis=1)
tumVeriler=np.concatenate([x_bias,y],axis=1)


egitimBoyutu=round(tumVeriler.shape[0]*0.8)
randomİndisler = np.random.choice(tumVeriler.shape[0], size=egitimBoyutu, replace=False)
egitimVerisi = tumVeriler[randomİndisler, :]
testVerisi = np.delete(tumVeriler, randomİndisler, axis=0)

x_egitim = egitimVerisi[:, :-1]
y_egitim = (egitimVerisi[:, -1]).reshape(egitimBoyutu, 1)
x_test = testVerisi[:, :-1]
y_test = (testVerisi[:, -1]).reshape(tumVeriler.shape[0]-egitimBoyutu, 1)

mada=Madaline() #aynı problemi çözmek için adaline yerine madaline sınıfını çağırdık.

mada.egit(x_egitim, y_egitim,0.00003,1000)
tahmin = mada.tahmin(x_egitim)

fig = plt.figure()
plt.plot(range(len(y_egitim)),y_egitim,c='r', label="gerçek değerler")
plt.plot(range(len(y_egitim)),tahmin,c='b', label="tahminler")
plt.legend(loc="upper left")
plt.title('2 boyutta, tahminlerimizin ve gerçek değerlerin ne kadar örtüştüğünü inceliyoruz.')
plt.show()
