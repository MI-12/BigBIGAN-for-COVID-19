import numpy as np
import h5py
import argparse
import imageio
import tqdm
import os
import cv2
import pydicom
import random
import torch
import skimage
from skimage import measure
import SimpleITK as itk
from glob import glob
from matplotlib import pyplot
from sklearn import preprocessing

def main(args):

    # create hdf5
    HDF5_Train = h5py.File(os.path.join(args.output_dir_train, "TrainData.hdf5"), "a")
    HDF5_Test = h5py.File(os.path.join(args.output_dir_train, "TestData.hdf5"), "a")
    HDF5_Va = h5py.File(os.path.join(args.output_dir_train, "TvalidationData.hdf5"), "a")

    # get all data directory
    list_dirs = os.listdir(args.input_dir_train)
    Counttrain = 0
    Counttest = 0
    Countva = 0
    for dirs in list_dirs:
        data_path_hosp = os.path.join(args.input_dir_train, dirs)
        data_train_Ori = glob(os.path.join(data_path_hosp,  "*"))

        length = len(data_train_Ori)
        index_train = int(round(length))
        index_test = int(round(length*0.9))
        indexes = np.array(range(0,length))
        random.shuffle(indexes)
        trn_idxes = indexes[0:index_train]
        tst_idxes = indexes[index_train:index_test]
        vad_idxes = indexes[index_test:length]

        with tqdm.tqdm(total=len(data_train_Ori), unit="folder") as progress_bar_train:
            for i, path_hosp in zip(indexes, data_train_Ori):
                print(i,data_train_Ori)
                data_name = path_hosp.split("/")[-1].split(".dcm")[0]

                x = parse_data(path_hosp)

                # TODO only use majority size for now
                if x is None:
                    progress_bar_train.update(1)
                    continue

                
                if i in trn_idxes:
                    HDF5_Train['%s/train' %data_name] = x
                    #HDF5_Train['%s/target' %data_name] = y
                    Counttrain = Counttrain + 1
                elif i in tst_idxes:
                    HDF5_Test['%s/train' %data_name] = x
                    HDF5_Test['%s/target' %data_name] = y
                    Counttest = Counttest + 1
                elif i in vad_idxes:
                    HDF5_Va['%s/train' %data_name] = x
                    HDF5_Va['%s/target' %data_name] = y
                    Countva = Countva + 1
                
                progress_bar_train.update(1)
    HDF5_Train.close()
    HDF5_Va.close()
    HDF5_Test.close()
    print(Counttrain)

def parse_data(path_Ori):

    x_dicom = pydicom.read_file(path_Ori)

    dcm = itk.ReadImage(path_Ori) 
    data_itk = itk.GetArrayFromImage(dcm)
    hist_np, _ = np.histogram(data_itk.ravel(),bins = 50) 
    
    if(hist_np[1] == 0): #Normalize images under different scanning conditions
        sort_data_itk = np.unique(np.sort(data_itk, axis=None)) 
        
        data_itk = np.where(data_itk == sort_data_itk[0], sort_data_itk[1], data_itk) 
        
        if(sort_data_itk[0] == -2048) | (sort_data_itk[0] == -2000):
            
            if(pydicom.read_file(path_Ori).RescaleIntercept == 0):
                data_itk = data_itk * pydicom.read_file(path_Ori).RescaleSlope + np.abs(sort_data_itk[1])
            else:
                data_itk = data_itk * pydicom.read_file(path_Ori).RescaleSlope + np.abs(pydicom.read_file(path_Ori).RescaleIntercept)
            #print(data_itk)  # -1024,unscaled, raw data
            #print(np.mean(data_itk), np.max(data_itk) - np.min(data_itk))
            if(np.max(data_itk) - np.min(data_itk) > 4000):
                data_itk = np.where(data_itk > 3000, 2700, data_itk)
            #print(np.mean(data_itk), np.max(data_itk) - np.min(data_itk))
            #print(pydicom.dcmread(path_Ori).pixel_array)  # == data_itk,unscaled, raw data
            #print(pydicom.read_file(path_Ori).RescaleSlope, np.abs(pydicom.read_file(path_Ori).RescaleIntercept))
            #print(np.mean(pydicom.read_file(path_Ori).pixel_array), np.max((pydicom.read_file(path_Ori).pixel_array))-np.min((pydicom.read_file(path_Ori).pixel_array)))
        else:
            if(pydicom.read_file(path_Ori).RescaleIntercept == 0):
                data_itk = data_itk * pydicom.read_file(path_Ori).RescaleSlope + np.abs(sort_data_itk[1])
            else:
                data_itk = data_itk * pydicom.read_file(path_Ori).RescaleSlope + np.abs(pydicom.read_file(path_Ori).RescaleIntercept)

            if(np.max(data_itk) - np.min(data_itk) > 4000):
                data_itk = np.where(data_itk > 3000, 2700, data_itk)
            
    else:    
        data_itk = data_itk * pydicom.read_file(path_Ori).RescaleSlope + np.abs(pydicom.read_file(path_Ori).RescaleIntercept) 
    data_itk = np.squeeze(data_itk)

    if x_dicom.pixel_array.shape != (256, 256):
        return None, None

    x = data_itk.astype(np.float32)
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1.0,1.0),copy = False)#范围改为1~3，对原数组操作
    x = min_max_scaler.fit_transform(x)
    print(np.max(x))
    x = np.expand_dims(x,2).repeat(3,axis=2)
    print(x.shape)
    return x

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir_train', type=str,default="Your Data Path")
    parser.add_argument('--output_dir_train', type=str,default="Your Data Path")
    parser.add_argument('--output_dir_test', type=str,default="Your Data Path")

    args = parser.parse_args()

    main(args)
