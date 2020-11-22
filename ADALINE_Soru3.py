import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand,randint
from numpy import cos,pi
from adaline_class import Adaline

x1= rand(50,1)
x2=(pi/2)*rand(50,1)
y= np.array([3*x1[i]+2*cos(x2[i]) for i in range(len(x1))])
x=np.concatenate([x1,x2],axis=1)
bias=np.ones((len(x),1))
x_bias=np.concatenate([x,bias],axis=1)
tumVeriler=np.concatenate([x_bias,y],axis=1)

# fig = plt.figure()
# ax=fig.add_subplot(111, projection='3d')
# ax.scatter3D(tumVeriler[:,0],tumVeriler[:,1],tumVeriler[:,3],'green')
# plt.show()

egitimBoyutu=round(tumVeriler.shape[0]*0.8)
randomİndisler = np.random.choice(tumVeriler.shape[0], size=egitimBoyutu, replace=False)
egitimVerisi = tumVeriler[randomİndisler, :]
testVerisi = np.delete(tumVeriler, randomİndisler, axis=0)

x_egitim = egitimVerisi[:, :-1]
y_egitim = (egitimVerisi[:, -1]).reshape(egitimBoyutu, 1)
x_test = testVerisi[:, :-1]
y_test = (testVerisi[:, -1]).reshape(tumVeriler.shape[0]-egitimBoyutu, 1)

ada=Adaline()

agirlikveCost=ada.egit(x_egitim, y_egitim,0.01,1000,0.01)
tahmin = ada.tahminEt(x_test)
cost=agirlikveCost[1]
sonİter=agirlikveCost[2]
print("eğitimimizi tamamladıktan sonraki ortalama kare hatamız: ",agirlikveCost[1][sonİter])

fig = plt.figure()
plt.plot(range(len(y_test)),y_test,c='r',label="gerçek değerler")
plt.plot(range(len(y_test)),tahmin,c='b', label="tahminler")
plt.legend(loc="upper left")
plt.title('2 boyutta, tahminlerimizin ve gerçek değerlerin ne kadar örtüştüğünü inceliyoruz.')

fig1 = plt.figure()
ax1=fig1.add_subplot(111, projection='3d')
ax1.scatter3D(x_test[:,0],x_test[:,1],y_test,'g', label="gerçek değerler")
ax1.scatter3D(x_test[:,0],x_test[:,1],tahmin,'b', label="tahminler")
plt.legend(loc="lower left")
plt.title('3 boyutta, tahminlerimizin ve gerçek değerlerin ne kadar örtüştüğünü inceliyoruz.')





plt.figure()
plt.plot(range(sonİter),agirlikveCost[1][:sonİter],c='b')
plt.xlabel("iterasyon sayısı")
plt.ylabel("Ortalama Kare Hata")

plt.show()