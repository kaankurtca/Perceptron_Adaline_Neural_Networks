import numpy as np
from numpy.random import rand,randint
from perceptron_class import GenlikteAyrikAlgilayici
import matplotlib.pyplot as plt


if __name__ == "__main__":

    a=np.array(randint(0,5,(40,4)))
    b=np.zeros((40,1))
    for ind in range(len(a)):
        if ((2*a[ind,0] - 3*a[ind,1] + 4*a[ind,2] - 3*a[ind,3]) >= 0):
            b[ind]=1
        else:
            b[ind]=-1
    #verisetinin lineer ayrıştırılabilir olması için 'y' sütununu bu şekilde oluşturduk.
    lineerAyristirilabilirDizi=np.concatenate([a,b],axis=1)


    toplamİter=np.zeros([20,1])
    accuracy=np.zeros([20,1])


    randomİndisler1 = np.random.choice(lineerAyristirilabilirDizi.shape[0], size=25, replace=False)
    egitimVerisi1=lineerAyristirilabilirDizi[randomİndisler1,:]
    testVerisi1=np.delete(lineerAyristirilabilirDizi,randomİndisler1,axis=0) #buradaki 3 satırda verimizi rastgele seçilen indisler ile eğitim ve test verisi ile ayırdık.

    x_egitim1 = egitimVerisi1[:,:-1]
    y_egitim1 = (egitimVerisi1[:,-1]).reshape(25,1)
    x_test1 = testVerisi1[:,:-1]
    y_test1 = (testVerisi1[:,-1]).reshape(15,1)

    gaa = GenlikteAyrikAlgilayici()
    for i in range(20):
        gaa.egit(x_egitim1,y_egitim1,0.1,100,1)
        toplamİter[i]=(gaa.egit(x_egitim1,y_egitim1,1,100,5))[1]
        tahmin=gaa.tahminEt(x_test1)
        sonuc=y_test1-tahmin

        accuracy[i]=gaa.skor(sonuc) #modelin doğru eğitilmiş olması için 1 skorunu bekliyoruz.
        #print("Liner ayrıştırılabilir verimizin skoru: ",accuracy1)

    plt.scatter(range(len(accuracy)),accuracy)
    plt.plot(range(len(accuracy)), accuracy)
    plt.show()

    plt.scatter(range(len(accuracy)), toplamİter)
    plt.plot(range(len(accuracy)), toplamİter)
    plt.show()
    

    gaa2=GenlikteAyrikAlgilayici()
    c=np.array(2*randint(0,2,[40,1])-1)
    lineerAyristirilamazDizi =np.concatenate([a,c],axis=1) #random sayılar seçerek lineer ayrıştırılamaz bir dizi oluşturduk.

    randomİndisler2 = np.random.choice(lineerAyristirilamazDizi.shape[0], size=25, replace=False)
    egitimVerisi2 = lineerAyristirilamazDizi[randomİndisler2, :]
    testVerisi2 = np.delete(lineerAyristirilamazDizi, randomİndisler2, axis=0)

    x_egitim2 = egitimVerisi2[:, :-1]
    y_egitim2 = (egitimVerisi2[:, -1]).reshape(25, 1)
    x_test2 = testVerisi2[:, :-1]
    y_test2 = (testVerisi2[:, -1]).reshape(15, 1)

    gaa2.egit(x_egitim2, y_egitim2, 1, 1000)
    print(gaa2.egit(x_egitim2, y_egitim2, 1, 1000))
    tahmin = gaa2.tahminEt(x_test2)
    sonuc = y_test2 - tahmin

    accuracy2 = gaa2.skor(sonuc)  # lineer ayrıştırılamaz veriseti olduğu için düşük bir skor bekliyoruz.
    print("Liner ayrıştırılamaz verimizin skoru: ", accuracy2)
