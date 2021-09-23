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
from train import epochVal
from dataset_loader import generate_dataset_loader
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

def window_image(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img

def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)

def save_img(img_pil, subfolder, name):
    img_pil.save(subfolder+'baturalp@'+name+'.png')

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)
def prepare_image(img_path,min_window,max_window):
    #img_id = get_id(img_dicom)
    img_id = ntpath.basename(img_path).replace(".dcm", "")
    itkimage = sitk.ReadImage(img_path)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyImage = numpyImage[0]
    img = window_image(numpyImage, min_window,max_window)
    img = normalize_minmax(img) * 255
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img_id, img

def prepare_and_save(img_path, subfolder):
    name_list = []
    for (dirpath, dirnames, filenames) in os.walk('./Baturalp_labels/Selected MASK/'):
        name_list.extend(filenames)
        break
    img_id = ntpath.basename(img_path).replace(".dcm", ".png")
    img_id = 'baturalp@'+img_id
    if img_id not in name_list:
        return
    try:
        img_id, img_pil = prepare_image(img_path,600,2800)
        save_img(img_pil, subfolder + 'extracted_png_bone/', img_id)
        img_id, img_pil = prepare_image(img_path,80,200)
        save_img(img_pil, subfolder + 'extracted_png_subdural/', img_id)
        img_id, img_pil, = prepare_image(img_path,40,80)
        save_img(img_pil, subfolder + 'extracted_png_brain/', img_id)
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


def prepare_data(Input,out_preprocessing_path):
    if not os.path.exists(out_preprocessing_path):
        os.makedirs(out_preprocessing_path)

    prepare_images_njobs(glob(Input + '/*'), out_preprocessing_path)
if __name__ == '__main__':
    #STEP1 :Preprocessing
    Input_path = "./Baturalp_labels/RSNA 1-500 DICOMS/"
    out_preprocessing_path = "./output/" #make sure it end with /
    prepare_data(Input_path,out_preprocessing_path)
    name_list = []
    for (dirpath, dirnames, filenames) in os.walk('./Baturalp_labels/Selected MASK/'):
        name_list.extend(filenames)
        break
    with open('output/'+ 'train_baturalp.txt', 'a', newline='') as f:
        writer = csv.writer(f)
        for cae in name_list:
            writer.writerow([cae])