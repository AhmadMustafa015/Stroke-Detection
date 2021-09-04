import ntpath
import random

import joblib
import PIL
from glob import glob
import pydicom
import numpy as np
import pandas as pd
import csv
import os
import cv2
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
import re
import logging as l
from glob import glob
import argparse
import settings
import SimpleITK as sitk
import pandas as pd
import pydicom.pixel_data_handlers.pillow_handler as dicom_pil
def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(img_dicom):
    return str(img_dicom.SOPInstanceUID)

def get_metadata_from_dicom(img_dicom):
    metadata = {
        "window_center": img_dicom.WindowCenter,
        "window_width": img_dicom.WindowWidth,
        "intercept": img_dicom.RescaleIntercept,
        "slope": img_dicom.RescaleSlope,
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

def window_image(img, window_center, window_width, intercept, slope):
    #img = img * slope + intercept #no need for this in simple itk
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img

def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)

def save_img(img_pil, subfolder, name,label):
    img_pil.save(subfolder+name+'_'+str(label)+'.png')

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)
def prepare_image(img_path):
    img_dicom = pydicom.read_file(img_path)
    #img_id = get_id(img_dicom)
    img_id = ntpath.basename(img_path).replace(".dcm", "")
    if "INMEYOK" in img_path:
        label = 0
    elif "ISKEMI" in img_path:
        label = 1
    elif "KANAMA" in img_path:
        label = 2
    else:
        assert False, ("Error wrong image label please check image path: ", img_path)
    itkimage = sitk.ReadImage(img_path)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyImage = numpyImage[0]
    metadata = get_metadata_from_dicom(img_dicom)
    img = window_image(numpyImage, **metadata)
    img = normalize_minmax(img) * 255
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img_id, img, label

def prepare_and_save(img_path, subfolder):
    try:
        img_id, img_pil,label = prepare_image(img_path)
        save_img(img_pil, subfolder, img_id, label)
    except KeyboardInterrupt:
         #Rais interrupt exception so we can stop the cell execution
         #without shutting down the kernel.
        raise
    except:
        l.error('Error processing the image: {'+img_path+'}')

def prepare_images(imgs_path, subfolder):
    for i in tqdm.tqdm(imgs_path):
        prepare_and_save(i, subfolder)

def prepare_images_njobs(img_paths, subfolder, n_jobs=-1):
    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(prepare_and_save)(i, subfolder) for i in tqdm(img_paths))
    #prepare_and_save(img_paths[0],subfolder)


def prepare_data():
    dcm_path = settings.dicom_path
    output_path = settings.png_out_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    prepare_images_njobs(glob(dcm_path + '/*/DICOM/*'), output_path + '/')
def get_metadata_info(window_center, window_width, intercept, slope):
    return window_center, window_width, intercept, slope
def prepare_csv():
    all_files = []
    directory = settings.png_out_path
    for filename in tqdm(os.listdir(directory)):
        img_path = settings.dicom_path
        if filename.endswith(".png"):
            parts = filename.split("_")
            class_label = int(parts[-1].replace(".png",""))
            patient_id = parts[-2]
            #print(patient_id, class_label)
            if class_label == 0:
                img_path += 'INMEYOK/DICOM/' + filename.replace("_0.png",".dcm")
                img_dicom = pydicom.read_file(img_path)
                metadata = get_metadata_from_dicom(img_dicom)
                window_center, window_width, intercept, slope = get_metadata_info(**metadata)
                all_files.append([patient_id,class_label,class_label,class_label,window_center, window_width, intercept, slope])
            elif class_label == 1:
                img_path += 'ISKEMI/DICOM/' + filename.replace("_1.png", ".dcm")
                img_dicom = pydicom.read_file(img_path)
                metadata = get_metadata_from_dicom(img_dicom)
                window_center, window_width, intercept, slope = get_metadata_info(**metadata)
                all_files.append([patient_id, class_label, class_label, 0,window_center, window_width, intercept, slope])
            elif class_label == 2:
                img_path += 'KANAMA/DICOM/' + filename.replace("_2.png", ".dcm")
                img_dicom = pydicom.read_file(img_path)
                metadata = get_metadata_from_dicom(img_dicom)
                window_center, window_width, intercept, slope = get_metadata_info(**metadata)
                all_files.append([patient_id, 1, 0, 1,window_center, window_width, intercept, slope])
    df = pd.DataFrame(all_files, columns=['filename', 'any', 'ISKEMI', 'KANAMA', 'window_center','window_width','rescale_intercept','rescale_slope'])
    df.to_csv('output/'+ "train.csv", index=False)

def fold_5_prepare():
    all_samples = []
    df = pd.read_csv('output/'+ "train.csv")
    for index, row in df.iterrows():
        all_samples.append(row['filename'])
    random.shuffle(all_samples)
    train_percent = int(len(all_samples) * 0.8)
    test_list = all_samples[train_percent:]
    with open('output/'+ 'test.txt', 'a', newline='') as f:
        writer = csv.writer(f)
        for cae in test_list:
            all_samples.remove(cae)
            writer.writerow([cae])
    for num_fold in range(5):
        random.shuffle(all_samples)
        train_percent = int(len(all_samples) * 0.8)
        train_data = all_samples[:train_percent]
        val_data = all_samples[train_percent:]
        with open('output/fold_5_by_study/fold'+str(num_fold)+'/train.txt', 'a', newline='') as f:
            writer = csv.writer(f)
            for cae in train_data:
                writer.writerow([cae])
        with open('output/fold_5_by_study/fold'+str(num_fold)+'/val.txt', 'a', newline='') as f:
            writer = csv.writer(f)
            for cae in val_data:
                writer.writerow([cae])

#prepare_csv()
fold_5_prepare()
#prepare_data()

