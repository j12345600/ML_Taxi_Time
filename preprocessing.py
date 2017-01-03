import csv
import numpy as np
import math
import os
import sys
import random
import scipy
import json
import datetime
# import utilities as utl

def RMSE(pre, real):
    npPre=np.array(pre)
    npReal=np.array(real)
    return math.sqrt(np.mean((npPre-npReal)*(npPre-npReal)))
def HaversineDistance(lat1,lon1,lat2,lon2):
    REarth=6371
    lat=-abs(lat1-lat2)*math.pi/180
    lon=-abs(lon1-lon2)*math.pi/180
    lat1=lat1*math.pi/180
    lat2=lat2*math.pi/180
    a=math.sin(lat/2)*math.sin(lat/2)+math.cos(lat1)*math.cos(lat2)*math.sin(lon/2)*math.sin(lon/2)
    d=2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d=REarth*d
    return d
def meanHaversineDistance(lat1,lon1,lat2,lon2):
    return(math.mean(HaversineDistance(lat1,lon1,lat2,lon2)))

def cal_dist_time(polyline):
    dist=time=0
    total_points=len(polyline)
    if total_points > 0:
        # time=(total_points-1)*15
        time=(total_points-1)/100
        for i in range(total_points-1):
            dist+=HaversineDistance(polyline[i][1],polyline[i][0],polyline[i+1][1],polyline[i+1][0])
    return dist,time
true_false_map_dict = {
    'True':1,
    'False':0
}
ABC_012_map_dict = {
    'A':0,
    'B':0.5,
    'C':1
}

def local_split(train_index):
   random.seed(0)
   train_index = set(train_index)
   all_index = sorted(train_index)
   num_test = int(len(all_index) / 4)
   random.shuffle(all_index)
   train_set = set(all_index[num_test:])
   test_set = set(all_index[:num_test])
   return train_set, test_set

def split_csv(src_csv, split_to_train, train_csv, test_csv):
   ftrain = open(train_csv, "w")
   ftest = open(test_csv, "w")
   cnt = 0
   for l in open(src_csv):
       if split_to_train[cnt]:
           ftrain.write(l)
       else:
           ftest.write(l)
       cnt = cnt + 1
   ftrain.close()
   ftest.close()
Path_train= "./data/train.csv"
Path_test=  "./data/test.csv"
Path_out=   "./procData/"
def timestampProc(tsmp):
    s=datetime.datetime.fromtimestamp(int(tsmp))\
    .strftime('%m,%d,%H')
    return s.split(',')
def gpsAngle(lat1,long1,lat2,long2):
    dy = lat2 - lat1
    dx = math.cos(math.pi/180*lat1)*(long2 - long1)
    return math.atan2(dy, dx)

def cal_poly_direction(poly):
    if len(poly) > 1:
        return gpsAngle(poly[0][1],poly[0][0],poly[-1][1],poly[-1][0])
    else:
        return 0
def write_trim_and_label_csv(fname,out):
    total=50000
    sub_total=10000
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        with open(out+"test_train_label.csv",'w', newline='') as w:
            with open(out+"test_train.csv",'w',newline='') as wt:
                writer_t=csv.writer(wt,quoting=csv.QUOTE_NONE)
                writer=csv.writer(w,quoting=csv.QUOTE_NONE)
                i=0
                for row in reader:
                    if i == 0:
                        i+=1
                        continue
                    poly=json.loads(row[8])
                    [dist,time]=cal_dist_time(poly)
                    if len(poly) > 0:
                        lat0=poly[0][1]
                        lon0=poly[0][0]
                        lat1=lon1=0
                        if len(poly) > 1:
                            lat1=poly[-1][1]
                            lon1=poly[-1][0]
                        else:
                            lat1=lat0
                            lon1=lon0
                    else:
                        lat0=lon0=lat1=lon1=0
                    mdH =timestampProc(row[5])
                    writer.writerow([time])
                    writer_t.writerow([ABC_012_map_dict[row[1]],(float(row[4])-20000000)/1000,float(mdH[0])/12,float(mdH[1])/30,float(mdH[2])/24,ABC_012_map_dict[row[6]],true_false_map_dict[row[7]],dist/10,lat0,lon0,cal_poly_direction(poly)])
                    i+=1
                    if i>total-sub_total:
                        break
        with open(out+"test_test_label.csv",'w', newline='') as w:
            with open(out+"test_test.csv",'w',newline='') as wt:
                writer_t=csv.writer(wt,quoting=csv.QUOTE_NONE)
                writer=csv.writer(w,quoting=csv.QUOTE_NONE)
                i=0
                for row in reader:
                    if i==0:
                        i+=1
                        continue
                    poly=json.loads(row[8])
                    [dist,time]=cal_dist_time(poly)
                    if len(poly) > 0:
                        lat0=poly[0][1]
                        lon0=poly[0][0]
                        lat1=lon1=0
                        if len(poly) > 1:
                            lat1=poly[-1][1]
                            lon1=poly[-1][0]
                        else:
                            lat1=lat0
                            lon1=lon0
                    else:
                        lat0=lon0=lat1=lon1=0
                    mdH =timestampProc(row[5])
                    writer.writerow([time])
                    writer_t.writerow([ABC_012_map_dict[row[1]],(float(row[4])-20000000)/1000,float(mdH[0])/12,float(mdH[1])/30,float(mdH[2])/24,ABC_012_map_dict[row[6]],true_false_map_dict[row[7]],dist/10,lat0,lon0,cal_poly_direction(poly)])
                    i+=1
                    if i>sub_total:
                        break
if __name__ == "__main__":
    write_trim_and_label_csv(Path_train,Path_out)
