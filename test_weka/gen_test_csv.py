import csv
import numpy as np
def FuncA(a,b,c):
    Max=max(a,b,c)
    if Max == a:
        return "a"
    elif Max == b:
        return "b"
    else:
        return "c"

mu, sigma = 0, 0.3 # mean and standard deviation
size=5000
sub_size=200
a = np.random.permutation(np.random.normal(mu, sigma, size))
b = np.random.permutation(np.random.normal(mu, sigma, size))
c = np.random.permutation(np.random.normal(mu, sigma, size))
Result=list()
with open('train.csv', 'w') as train:
    tw = csv.writer(train,quotechar='|', quoting=csv.QUOTE_MINIMAL)
    tw.writerow(["a[i]","b[i]","c[i]","result"])
    for i in range(size-sub_size):
        result=FuncA(a[i],b[i],c[i])
        tw.writerow([a[i],b[i],c[i],result])
with open('test.csv', 'w') as test:
    tsw = csv.writer(test,quotechar='|', quoting=csv.QUOTE_MINIMAL)
    tsw.writerow(["a[i]","b[i]","c[i]","result"])
    for i in range(size-sub_size,size):
        result=FuncA(a[i],b[i],c[i])
        tsw.writerow([a[i],b[i],c[i],result])
