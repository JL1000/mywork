#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import numpy as np  
from keras.preprocessing import *
from keras import backend as K 
from keras.models import load_model
def normlized(data ): 
    x_mean = np.array([
    83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
    66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
    0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
    22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
    0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
    4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
    38.9974, 10.5585,  286.5404, 198.6777])
    x_std = np.array([
        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
        29.8928, 7.0606,  137.3886, 96.8997])
    c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
    c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

    x = data[:, 0:34]
    c = data[:, 34:40]
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    c_norm = np.nan_to_num((c - c_mean) / c_std)
    dataArray = np.concatenate((x_norm, c_norm), axis=1)
    return dataArray  
def get_sepsis_score(values, model): 
    uV=normlized(np.array([values[-1]])) 
    prediction = model.predict(np.array([uV]))   
    if prediction[0][1]<0.45: 
        label=0
    else:
        label=1                  
    return prediction[0][1], label
def load_sepsis_model():
    return load_model("./myModel.h5" ) 