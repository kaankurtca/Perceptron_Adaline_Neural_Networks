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
        gaa.egit(x_egitim1,y_egitim1,1,100,1)
        toplamİter[i]=(gaa.egit(x_egitim1,y_egitim1,1,100,5))[1]
        tahmin=gaa.tahminEt(x_test1)
        sonuc=y_test1-tahmin
        accuracy[i]=gaa.skor(sonuc) #modelin doğru eğitilmiş olması için 1 skorunu bekliyoruz.
    plt.figure()
    plt.scatter(range(1,len(accuracy)+1),accuracy)
    plt.plot(range(1,len(accuracy)+1), accuracy)

    plt.figure()
    plt.scatter(range(1,len(toplamİter)+1), toplamİter)
    plt.plot(range(1,len(toplamİter)+1), toplamİter)



    c=np.array(2*randint(0,2,[40,1])-1)
    lineerAyristirilamazDizi = np.concatenate([x_bias,c],axis=1) #random sayılar seçerek lineer ayrıştırılamaz bir dizi oluşturduk.



    randomİndisler2 = np.random.choice(lineerAyristirilamazDizi.shape[0], size=25, replace=False)
    egitimVerisi2 = lineerAyristirilamazDizi[randomİndisler2, :]
    testVerisi2 = np.delete(lineerAyristirilamazDizi, randomİndisler2, axis=0)

    x_egitim2 = egitimVerisi2[:, :-1]
    y_egitim2 = (egitimVerisi2[:, -1]).reshape(25, 1)
    x_test2 = testVerisi2[:, :-1]
    y_test2 = (testVerisi2[:, -1]).reshape(15, 1)

    toplamİter2 = np.zeros([20, 1])
    accuracy2 = np.zeros([20, 1])

    gaa2=GenlikteAyrikAlgilayici()
    for i in range(20):
        gaa2.egit(x_egitim2, y_egitim2, 1, 100, 1)
        toplamİter2[i] = (gaa2.egit(x_egitim2, y_egitim2, 1, 100, 5))[1]
        tahmin2 = gaa2.tahminEt(x_test2)
        sonuc = y_test2 - tahmin2
        accuracy2[i] = gaa.skor(sonuc)  # modelin doğru eğitilmiş olması için 1 skorunu bekliyoruz.
    plt.figure()
    plt.scatter(range(1, len(accuracy2) + 1), accuracy2)
    plt.plot(range(1, len(accuracy2) + 1), accuracy2)

    plt.figure()
    plt.scatter(range(1, len(toplamİter2) + 1), toplamİter2)
    plt.plot(range(1, len(toplamİter2) + 1), toplamİter2)
    plt.show()






    #print("Liner ayrıştırılamaz verimizin skoru: ", accuracy2)
