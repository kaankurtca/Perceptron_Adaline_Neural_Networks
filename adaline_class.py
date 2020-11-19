import numpy as np
from numpy.random import rand,randint

class Adaline():
    def __init__(self):
        self.agirlik = [] #ağırlığımızı burada boş bir dizi olarak tanımladık.

    def aktivasyonFonks(self, x):
        return 1 / (1 + np.exp(-x))

    def aktivasyonTurev(self, x):
        return x * (1 - x)

    def egit(self,X,y,n_iterasyon,n):
        if n==0:
            self.agirlik = np.zeros(X.shape[1]).reshape(-1, 1)  # ağırlık vektörümüzü başlangıçta 0 olarak seçtik.
        elif n==1:
            self.agirlik = np.ones(X.shape[1]).reshape(-1, 1)  # ağırlık vektörümüzü başlangıçta 1 olarak seçtik.
        else:
            self.agirlik = np.random.rand([X.shape[0],1])  # ağırlık vektörümüzü başlangıçta ranstgele seçtik.

        for i in range(n_iterasyon):
            for j in range(X.shape[0]):
                tahmin = self.tahminEt(X[j]) #aşağıda tanımladığımız tahminEt() methodunu burada modelimizin eğitimi için kullandık.
                hata=y[j]-tahmin
                delta = (hata) * self.aktivasyonTurev(tahmin)
                self.agirlik += (delta * X[j].reshape(-1,1)) #tahmin ve gerçek değer arasındaki farklar ile ağırlığımızı her döngüde değiştiriyoruz.


        return self.agirlik

    def tahminEt(self,input):
        toplam=np.dot(input,self.agirlik) #girdilerimiz ve ağırlıklar çarpılarak toplam değeri elde ediliyor.
        output = np.array([self.aktivasyonFonks(toplam[ind]) for ind in range(toplam.shape[0])])

        return output

    def ortKareHata(self,tahmin,y):
        toplamKareHata= np.array([(y[i]-tahmin[i])**2 for i in range(tahmin.shape[0])])
        return sum(toplamKareHata)/tahmin.shape[0]