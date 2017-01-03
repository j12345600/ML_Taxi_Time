import csv
import numpy as np
def randomNumber(a,b):
    return ((b - a) * np.random.random_sample() + a)
def FuncA(a,b,c):
    res=2*a+4*b-5*c
    return res
    # return 0 if res > 0 else 1
    # Max=max(a,b,c)
    # if Max == a:
    #     return 0
    # elif Max == b:
    #     return 1
    # else:
    #     return 2

mu, sigma = 0, 0.3 # mean and standard deviation
size=1000
sub_size=100
a = np.random.permutation(np.random.normal(mu, sigma, size))
b = np.random.permutation(np.random.normal(mu, sigma, size))
c = np.random.permutation(np.random.normal(mu, sigma, size))
Result=list()
with open('train.csv', 'w') as train:
    tw = csv.writer(train,quotechar='|', quoting=csv.QUOTE_MINIMAL)
    with open('train_label.csv', 'w') as tlabel:
        tlw = csv.writer(tlabel,quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(size-sub_size):
            result=FuncA(a[i],b[i],c[i])
            tw.writerow([a[i],b[i],c[i]])
            tlw.writerow([result])
with open('test.csv', 'w') as test:
    tsw = csv.writer(test,quotechar='|', quoting=csv.QUOTE_MINIMAL)
    with open('test_label.csv', 'w') as tslabel:
        tslw = csv.writer(tslabel,quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(size-sub_size,size):
            result=FuncA(a[i],b[i],c[i])
            tsw.writerow([a[i],b[i],c[i]])
            tslw.writerow([result])
