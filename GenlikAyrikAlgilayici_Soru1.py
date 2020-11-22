import numpy as np
from numpy.random import rand,randint
from perceptron_class import GenlikteAyrikAlgilayici
import matplotlib.pyplot as plt


if __name__ == "__main__":

    x=np.array(randint(0,5,(40,4)))
    bias = np.ones((len(x), 1))
    x_bias=np.concatenate([x,bias],axis=1)
    y=np.zeros((40,1))
    for ind in range(len(x_bias)):
        if ((2*x[ind,0] - 3*x[ind,1] + 4*x[ind,2] - 3*x[ind,3]) >= 0):
            y[ind]=1
        else:
            y[ind]=-1
    #verisetinin lineer ayrıştırılabilir olması için 'y' sütununu bu şekilde oluşturduk.

    lineerAyristirilabilirDizi=np.concatenate([x_bias,y],axis=1)

    gaa = GenlikteAyrikAlgilayici()

    toplamİter=np.zeros([20,1])
    accuracy=np.zeros([20,1])

    for i in range(20):
        randomİndisler1 = np.random.choice(lineerAyristirilabilirDizi.shape[0], size=25, replace=False)
        egitimVerisi1=lineerAyristirilabilirDizi[randomİndisler1,:]
        testVerisi1=np.delete(lineerAyristirilabilirDizi,randomİndisler1,axis=0) #buradaki 3 satırda verimizi rastgele seçilen indisler ile eğitim ve test verisi olarak ayırdık.

        x_egitim1 = egitimVerisi1[:,:-1]
        y_egitim1 = (egitimVerisi1[:,-1]).reshape(25,1)
        x_test1 = testVerisi1[:,:-1]
        y_test1 = (testVerisi1[:,-1]).reshape(15,1)



        gaa.egit(x_egitim1,y_egitim1,0.1,1000,1)
        toplamİter[i]=(gaa.egit(x_egitim1,y_egitim1,0.1,1000,1))[1]
        tahmin=gaa.tahminEt(x_test1)
        sonuc=y_test1-tahmin
        accuracy[i]=gaa.skor(sonuc)
        #modelimizi 20 kez farklı eğitim kümeleriyle eğittik ve skorları bir dizide tuttuk


    plt.figure()
    plt.scatter(range(1,len(accuracy)+1),accuracy)
    plt.plot(range(1,len(accuracy)+1), accuracy, label="Lineer veriseti için, 20 farklı egitim sonucu elde edilen skorlar")
    plt.legend(loc="upper left")

    acc = sum(accuracy) / len(accuracy)
    print("Ayrıştırılabilir dizide, 20 eğitim sonucu test verimizdeki ortalama doğruluk skorumuz: ", acc)

    plt.figure()
    plt.scatter(range(1,len(toplamİter)+1), toplamİter)
    plt.plot(range(1,len(toplamİter)+1), toplamİter, label="Lineer veriseti için, 20 farklı eğitimde Modelin eğitilmesi için yapılan toplam iterasyon sayısı")
    plt.legend(loc="upper left")



    c=np.array(2*randint(0,2,[40,1])-1)
    lineerAyristirilamazDizi = np.concatenate([x_bias,c],axis=1) #random sayılar seçerek lineer ayrıştırılamaz bir dizi oluşturduk.

    gaa2 = GenlikteAyrikAlgilayici()

    toplamİter2 = np.zeros([20, 1])
    accuracy2 = np.zeros([20, 1])

    for i in range(20):
        randomİndisler2 = np.random.choice(lineerAyristirilamazDizi.shape[0], size=25, replace=False)
        egitimVerisi2 = lineerAyristirilamazDizi[randomİndisler2, :]
        testVerisi2 = np.delete(lineerAyristirilamazDizi, randomİndisler2, axis=0)

        x_egitim2 = egitimVerisi2[:, :-1]
        y_egitim2 = (egitimVerisi2[:, -1]).reshape(25, 1)
        x_test2 = testVerisi2[:, :-1]
        y_test2 = (testVerisi2[:, -1]).reshape(15, 1)

        gaa2.egit(x_egitim2, y_egitim2, 1, 100, 1)
        toplamİter2[i] = (gaa2.egit(x_egitim2, y_egitim2, 1, 100, 5))[1]
        tahmin2 = gaa2.tahminEt(x_test2)
        sonuc = y_test2 - tahmin2
        accuracy2[i] = gaa.skor(sonuc)

    plt.figure()
    plt.scatter(range(1, len(accuracy2) + 1), accuracy2)
    plt.plot(range(1, len(accuracy2) + 1), accuracy2, label="Non-Lineer veriseti için, 20 farklı egitim sonucu elde edilen skorlar")
    plt.legend(loc="upper left")

    acc2 = sum(accuracy2) / len(accuracy2)
    print("Ayrıştırılamayan dizide, 20 eğitim sonucu test verimizdeki ortalama doğruluk skorumuz: ", acc2)


    plt.figure()
    plt.scatter(range(1, len(toplamİter2) + 1), toplamİter2)
    plt.plot(range(1, len(toplamİter2) + 1), toplamİter2, label="Non-Lineer veriseti için, 20 farklı eğitimde Modelin eğitilmesi için yapılan toplam iterasyon sayısı")
    plt.ticklabel_format(useOffset=False)
    plt.legend(loc="upper left")
    plt.title('veriseti nonlineer olduğundan son iterasyona kadar eğitilmemesini beklyioruz')
    plt.show()

