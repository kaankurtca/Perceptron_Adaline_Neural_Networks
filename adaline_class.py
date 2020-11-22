import numpy as np
from numpy.random import rand,randint

class Adaline():
    def __init__(self):
        self.w_ = [] #ağırlığımızı burada boş bir dizi olarak tanımladık.

    def aktivasyonFonks(self, x):

        return x

    def aktivasyonTurev(self, x):
        return 1

    def egit(self,X,y,lr,n_iterasyon,eps):

        self.w_ = np.ones(X.shape[1]).reshape(-1, 1)  # ağırlık vektörümüzü başlangıçta 1 olarak seçtik.


        cost=np.zeros((n_iterasyon,1))
        for i in range(n_iterasyon):
            for j in range(X.shape[0]):
                tahmin = self.tahminEt(X[j]) #aşağıda tanımladığımız tahminEt() methodunu burada modelimizin eğitimi için kullandık.
                hata=y[j]-tahmin
                delta = lr *(hata) * self.aktivasyonTurev(tahmin)
                self.w_ += (delta * X[j].reshape(-1,1)) #tahmin ve gerçek değer arasındaki farklar ile ağırlığımızı her döngüde değiştiriyoruz.

            cost[i]=sum(0.5*((y-self.tahminEt(X))**2)/len(y))
            if cost[i]<eps:
                sonİter=i
                break
            else:
                sonİter=i


        return [self.w_,cost,sonİter]

    def tahminEt(self,input):
        toplam=np.dot(input,self.w_) #girdilerimiz ve ağırlıklar çarpılarak toplam değeri elde ediliyor.
        output = np.array([self.aktivasyonFonks(toplam[ind]) for ind in range(toplam.shape[0])])

        return output

