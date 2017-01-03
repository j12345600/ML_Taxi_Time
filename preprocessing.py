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
weather_map_int_dict = {
        "NULL"                          :0,
        ""                              :0,
        "Clear"                         :1,
        "Heavy Rain Showers"            :2,
        "Heavy Thunderstorms and Rain"  :3,
        "Light Drizzle"                 :4,
        "Light Rain"                    :5,
        "Light Rain Showers"            :6,
        "Light Thunderstorms and Rain"  :7,
        "Mostly Cloudy"                 :8,
        "Overcast"                      :9,
        "Partly Cloudy"                 :10,
        "Rain"                          :11,
        "Rain Showers"                  :12,
        "Scattered Clouds"              :13,
        "Thunderstorm"                  :14,
        "Thunderstorms and Rain"        :15
        }
time_map_dict = {
        "00:04":0,
        "00:09":1,
        "00:14":2,
        "00:19":3,
        "00:24":4,
        "00:29":5,
        "00:34":6,
        "00:39":7,
        "00:44":8,
        "00:49":9,
        "00:54":10,
        "00:59":11,
        "01:04":12,
        "01:09":13,
        "01:14":14,
        "01:19":15,
        "01:24":16,
        "01:29":17,
        "01:34":18,
        "01:39":19,
        "01:44":20,
        "01:49":21,
        "01:54":22,
        "01:59":23,
        "02:04":24,
        "02:09":25,
        "02:14":26,
        "02:19":27,
        "02:24":28,
        "02:29":29,
        "02:34":30,
        "02:39":31,
        "02:44":32,
        "02:49":33,
        "02:54":34,
        "02:59":35,
        "03:04":36,
        "03:09":37,
        "03:14":38,
        "03:19":39,
        "03:24":40,
        "03:29":41,
        "03:34":42,
        "03:39":43,
        "03:44":44,
        "03:49":45,
        "03:54":46,
        "03:59":47,
        "04:04":48,
        "04:09":49,
        "04:14":50,
        "04:19":51,
        "04:24":52,
        "04:29":53,
        "04:34":54,
        "04:39":55,
        "04:44":56,
        "04:49":57,
        "04:54":58,
        "04:59":59,
        "05:04":60,
        "05:09":61,
        "05:14":62,
        "05:19":63,
        "05:24":64,
        "05:29":65,
        "05:34":66,
        "05:39":67,
        "05:44":68,
        "05:49":69,
        "05:54":70,
        "05:59":71,
        "06:04":72,
        "06:09":73,
        "06:14":74,
        "06:19":75,
        "06:24":76,
        "06:29":77,
        "06:34":78,
        "06:39":79,
        "06:44":80,
        "06:49":81,
        "06:54":82,
        "06:59":83,
        "07:04":84,
        "07:09":85,
        "07:14":86,
        "07:19":87,
        "07:24":88,
        "07:29":89,
        "07:34":90,
        "07:39":91,
        "07:44":92,
        "07:49":93,
        "07:54":94,
        "07:59":95,
        "08:04":96,
        "08:09":97,
        "08:14":98,
        "08:19":99,
        "08:24":100,
        "08:29":101,
        "08:34":102,
        "08:39":103,
        "08:44":104,
        "08:49":105,
        "08:54":106,
        "08:59":107,
        "09:04":108,
        "09:09":109,
        "09:14":110,
        "09:19":111,
        "09:24":112,
        "09:29":113,
        "09:34":114,
        "09:39":115,
        "09:44":116,
        "09:49":117,
        "09:54":118,
        "09:59":119,
        "10:04":120,
        "10:09":121,
        "10:14":122,
        "10:19":123,
        "10:24":124,
        "10:29":125,
        "10:34":126,
        "10:39":127,
        "10:44":128,
        "10:49":129,
        "10:54":130,
        "10:59":131,
        "11:04":132,
        "11:09":133,
        "11:14":134,
        "11:19":135,
        "11:24":136,
        "11:29":137,
        "11:34":138,
        "11:39":139,
        "11:44":140,
        "11:49":141,
        "11:54":142,
        "11:59":143,
        "12:04":144,
        "12:09":145,
        "12:14":146,
        "12:19":147,
        "12:24":148,
        "12:29":149,
        "12:34":150,
        "12:39":151,
        "12:44":152,
        "12:49":153,
        "12:54":154,
        "12:59":155,
        "13:04":156,
        "13:09":157,
        "13:14":158,
        "13:19":159,
        "13:24":160,
        "13:29":161,
        "13:34":162,
        "13:39":163,
        "13:44":164,
        "13:49":165,
        "13:54":166,
        "13:59":167,
        "14:04":168,
        "14:09":169,
        "14:14":170,
        "14:19":171,
        "14:24":172,
        "14:29":173,
        "14:34":174,
        "14:39":175,
        "14:44":176,
        "14:49":177,
        "14:54":178,
        "14:59":179,
        "15:04":180,
        "15:09":181,
        "15:14":182,
        "15:19":183,
        "15:24":184,
        "15:29":185,
        "15:34":186,
        "15:39":187,
        "15:44":188,
        "15:49":189,
        "15:54":190,
        "15:59":191,
        "16:04":192,
        "16:09":193,
        "16:14":194,
        "16:19":195,
        "16:24":196,
        "16:29":197,
        "16:34":198,
        "16:39":199,
        "16:44":200,
        "16:49":201,
        "16:54":202,
        "16:59":203,
        "17:04":204,
        "17:09":205,
        "17:14":206,
        "17:19":207,
        "17:24":208,
        "17:29":209,
        "17:34":210,
        "17:39":211,
        "17:44":212,
        "17:49":213,
        "17:54":214,
        "17:59":215,
        "18:04":216,
        "18:09":217,
        "18:14":218,
        "18:19":219,
        "18:24":220,
        "18:29":221,
        "18:34":222,
        "18:39":223,
        "18:44":224,
        "18:49":225,
        "18:54":226,
        "18:59":227,
        "19:04":228,
        "19:09":229,
        "19:14":230,
        "19:19":231,
        "19:24":232,
        "19:29":233,
        "19:34":234,
        "19:39":235,
        "19:44":236,
        "19:49":237,
        "19:54":238,
        "19:59":239,
        "20:04":240,
        "20:09":241,
        "20:14":242,
        "20:19":243,
        "20:24":244,
        "20:29":245,
        "20:34":246,
        "20:39":247,
        "20:44":248,
        "20:49":249,
        "20:54":250,
        "20:59":251,
        "21:04":252,
        "21:09":253,
        "21:14":254,
        "21:19":255,
        "21:24":256,
        "21:29":257,
        "21:34":258,
        "21:39":259,
        "21:44":260,
        "21:49":261,
        "21:54":262,
        "21:59":263,
        "22:04":264,
        "22:09":265,
        "22:14":266,
        "22:19":267,
        "22:24":268,
        "22:29":269,
        "22:34":270,
        "22:39":271,
        "22:44":272,
        "22:49":273,
        "22:54":274,
        "22:59":275,
        "23:04":276,
        "23:09":277,
        "23:14":278,
        "23:19":279,
        "23:24":280,
        "23:29":281,
        "23:34":282,
        "23:39":283,
        "23:44":284,
        "23:49":285,
        "23:54":286,
        "23:59":287
        }
def mkdir(fname):
   try:
       os.mkdir(fname)
   except:
       pass

def get_label_map(fname):
   labelmap = {}
   fi = open(fname)
   fi.readline()
   for line in fi:
       arr = line.split(',')
       labelmap[int(arr[0])] = line
   return labelmap

def write_label_csv(fname, frames, label_map):
   fo = open(fname, "w")
   for lst in frames:
       index = int(lst[0].split("/")[3])
       if label_map != None:
           fo.write(label_map[index])
       else:
           fo.write("%d,0,0\n" % index)
   fo.close()

def write_data_csv(fname, frames, preproc):
   """Write data to csv file"""
   fdata = open(fname, "w")
   dr = Parallel()(delayed(get_data)(lst,preproc) for lst in frames)
   data,result = zip(*dr)
   for entry in data:
      fdata.write(','.join(entry)+'\r\n')
   print("All finished, %d slices in total" % len(data))
   fdata.close()
   result = np.ravel(result)
   return result

def crop_resize(img, size):
   """crop center and resize"""
   if img.shape[0] < img.shape[1]:
       img = img.T
   # we crop image from center
   short_egde = min(img.shape[:2])
   yy = int((img.shape[0] - short_egde) / 2)
   xx = int((img.shape[1] - short_egde) / 2)
   crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
   # resize to 64, 64
   resized_img = transform.resize(crop_img, (size, size))
   resized_img *= 255
   return resized_img.astype("uint8")

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
if(False):
    # Load the list of all the training frames, and shuffle them
    # Shuffle the training frames
    random.seed(10)
    train_frames = get_frames("./data/train")
    random.shuffle(train_frames)
    validate_frames = get_frames("./data/validate")

    # Write the corresponding label information of each frame into file.
    write_label_csv("./train-label.csv", train_frames, get_label_map("./data/train.csv"))
    write_label_csv("./validate-label.csv", validate_frames, None)

    # Dump the data of each frame into a CSV file, apply crop to 64 preprocessor
    train_lst = write_data_csv("./train-64x64-data.csv", train_frames, lambda x: crop_resize(x, 64))
    valid_lst = write_data_csv("./validate-64x64-data.csv", validate_frames, lambda x: crop_resize(x, 64))

    # Generate local train/test split, which you could use to tune your model locally.
    train_index = np.loadtxt("./train-label.csv", delimiter=",")[:,0].astype("int")
    train_set, test_set = local_split(train_index)
    split_to_train = [x in train_set for x in train_index]
    split_csv("./train-label.csv", split_to_train, "./local_train-label.csv", "./local_test-label.csv")
    split_csv("./train-64x64-data.csv", split_to_train, "./local_train-64x64-data.csv", "./local_test-64x64-data.csv")
Path_train= "./data/train.csv"
Path_test=  "./data/test.csv"
Path_out=   "./procData/"
def class_available(total):
    empty=1
    low=6
    enough=10
    if(total>40):
        low=int(total/7.0)
        enough=int(total/4.0)
    elif (total>60):
        low=int(60/7.0)
        enough=int(60/4.0)
        empty=2
    elif (total>80):
        low=int(60/7.0)
        enough=int(60/4.0)
        empty=3
    return (empty,low,enough)
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
if(False):
    total_space=utl.get_total(Path_train)
    write_trim_and_label_csv(Path_train,total_space)
    write_trim_and_label_csv(Path_test,total_space)

    train_index = list(range(0,utl.csv_shape(Path_base + "train61_label.csv")[0]))
    train_set, test_set = local_split(train_index)
    split_to_train = [x in train_set for x in train_index]
    split_csv(Path_base + "train61_trim.csv", split_to_train, Path_base + "train.csv", Path_base + "val.csv")
    split_csv(Path_base + "train61_label.csv", split_to_train, Path_base + "train_label.csv", Path_base + "val_label.csv")

write_trim_and_label_csv(Path_train,Path_out)
