import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from perceptron_class import GenlikteAyrikAlgilayici

dizi=np.array([[0,-1,1],[0,0,1],[0,1,1],[1,-1,1],[1,0,1],[1,1,1],[-1,-1,1],[-1,0,1],[-1,1,1],[-3,3,-1],[-3,1,-1],[-3,0,-1],[-3,-1,-1],[-3,-3,-1],[-1,3,-1],[-1,-3,-1],[0,3,-1],[0,-3,-1],[1,3,-1],[1,-3,-1],[3,3,-1],[3,1,-1],[3,0,-1],[3,-1,-1],[3,-3,-1],[-2,3,-1],[-3,2,-1],[-3,-2,-1],[-2,-3,-1],[2,3,-1],[3,2,-1],[3,-2,-1],[2,-3,-1]])

X=dizi[:,:-1]
y=dizi[:,-1].reshape(33,1)

for i in range(X.shape[0]):
    if y[i]==1:
        plt.scatter(X[i,0],X[i,1],c='r')
    else:
        plt.scatter(X[i,0],X[i,1],c='g')



eklenenSutun=np.ones([X.shape[0],1])
for i in range(X.shape[0]):
    eklenenSutun[i]=  X[i,0]**2+X[i,1]**2         #X[i,0]**2 + X[i,1]**2          #y[i] - (3*X[i,0]+8*X[i,1])/2 + 0.25*np.random.rand()

# X[:,0]= [X[index,0]+X[index,1] for index in range(len(X))]
# X[:,1]= [X[index,1]-X[index,1] for index in range(len(X))]
# eklenenSutun=np.ones([X.shape[0],1])
# eklenenSutun = np.array([X[index,0]**2 for index in range(len(X))])

X=np.concatenate([X,eklenenSutun],axis=1)
bias=np.ones((len(X),1))
X_bias=np.concatenate([X,bias],axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(X.shape[0]):
    if y[i]==1:
        ax.scatter(X[i,0],X[i,1],X[i,2],c='r')
    else:
        ax.scatter(X[i,0],X[i,1],X[i,2],c='g')


tumVeriler=np.concatenate([X_bias,y],axis=1)

egitimBoyutu=round(tumVeriler.shape[0]*0.8)
randomİndisler = np.random.choice(tumVeriler.shape[0], size=egitimBoyutu, replace=False)
egitimVerisi = tumVeriler[randomİndisler, :]
testVerisi = np.delete(tumVeriler, randomİndisler, axis=0)

x_egitim = egitimVerisi[:, :-1]
y_egitim = (egitimVerisi[:, -1]).reshape(egitimBoyutu, 1)
x_test = testVerisi[:, :-1]
y_test = (testVerisi[:, -1]).reshape(tumVeriler.shape[0]-egitimBoyutu, 1)

gaa=GenlikteAyrikAlgilayici()

gaa.egit(x_egitim, y_egitim, 0.1, 100,1)
tahmin = gaa.tahminEt(x_test)
sonuc = y_test - tahmin
accuracy = gaa.skor(sonuc)

print(accuracy)
print((gaa.egit(x_egitim, y_egitim, 1, 100,1))[1])

plt.show()
