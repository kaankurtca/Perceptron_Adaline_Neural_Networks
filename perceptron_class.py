import numpy as np
from numpy.random import rand,randint

class GenlikteAyrikAlgilayici():

    def __init__(self):
        self.agirlik = [] #ağırlığımızı burada boş bir dizi olarak tanımladık.

    def aktivasyonFonks(self, x):
        return 1 if x>=0 else -1  #değerleri 1 veya -1'e eşitleyen aktivasyon fonksiyonu (sgn(x))

    def egit(self,X,y,ogrHizi,n_iterasyon,n): #
        if n == 0:
            self.agirlik = np.zeros(X.shape[1]).reshape(-1, 1)  # ağırlık vektörümüzü başlangıçta 0 olarak seçtik.
        elif n == 1:
            self.agirlik = np.ones(X.shape[1]).reshape(-1, 1)  # ağırlık vektörümüzü başlangıçta 1 olarak seçtik.
        else:
            self.agirlik = np.random.rand(X.shape[1], 1).reshape(-1, 1)   # ağırlık vektörümüzü başlangıçta ranstgele seçtik.

        for i in range(n_iterasyon):
            for j in range(X.shape[0]):
                tahmin = self.tahminEt(X[j]) #aşağıda tanımladığımız tahminEt() methodunu burada modelimizin eğitimi için kullandık.
                delta = ogrHizi * (y[j]-tahmin)
                self.agirlik += (delta * X[j].reshape(-1,1)) #tahmin ve gerçek değer arasındaki farklar ile ağırlığımızı her döngüde değiştiriyoruz.
            if self.skor(y-self.tahminEt(X))==1:

                toplamİter=i
                break
        return [self.agirlik,toplamİter]

    def tahminEt(self,input):
        toplam=np.dot(input,self.agirlik) #girdilerimiz ve ağırlıklar çarpılarak toplam değeri elde ediliyor.
        output=np.zeros(toplam.shape[0]).reshape((-1,1))
        for ind in range(toplam.shape[0]):
            output[ind] = self.aktivasyonFonks(toplam[ind]) #toplam vektörünün içindeki değerleri tek tek akt.fonk. sokarak çıktımızı elde ettik.
        return output

    def skor(self,sonuc):
        correct=0
        for s in sonuc:
            if s == 0:
                correct+=1
        return correct/sonuc.shape[0]