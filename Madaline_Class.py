import numpy as np
from numpy.random import rand,randint

class Madaline():
    def __init__(self):
        self.w_ = [] #ağırlığımızı burada boş bir dizi olarak tanımladık.

    def egit(self,X,y,lr,n_iterasyon):

        self.w_ = 0.4*np.ones((2,X.shape[1]))
        self.c_= 0.4*np.ones((1,3))

        for i in range(n_iterasyon):
            for j in range(X.shape[0]):
                tahmin1=np.zeros((2,1))
                tahmin1= self.tahminEtİlkKatman(X[j])
                tahmin1=np.append(tahmin1,1).reshape(-1,1)
                tahmin2=self.tahminEtİkinciKatman(tahmin1)

                if y[j]!=tahmin2:
                    for k in range(self.w_.shape[0]):
                        for l in range(self.w_.shape[1]):
                            self.w_[k,l] += lr*(y[j]-tahmin1[l])*X[j,k]
                            #Madaline yapısının öğrenme kuralı uygulandı
        return self.w_

    def tahminEtİlkKatman(self,input):
        toplam1=np.ones((2,1))
        toplam1=np.dot(self.w_,input.reshape(-1,1)) #girdilerimiz ve ağırlıklar çarpılarak toplam değeri elde ediliyor.
        # output = np.array([self.aktivasyonFonks(toplam[ind]) for ind in range(toplam.shape[0])])
        return toplam1
    def tahminEtİkinciKatman(self,input):
        toplam2=np.dot(self.c_,input)
        return toplam2

    def tahmin(self,input):
        sonuc1=np.zeros((2,len(input)))
        sonuc1=np.dot(self.w_,input.T)
        biass=np.ones((1,sonuc1.shape[1]))
        z=np.concatenate([sonuc1,biass],axis=0)
        sonuc2=np.zeros((1,z.shape[1]))
        sonuc2=np.dot(self.c_,z).reshape(-1,1)

        return sonuc2

