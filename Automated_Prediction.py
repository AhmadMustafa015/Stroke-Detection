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

def save_img(img_pil, subfolder, name,label):
    img_pil.save(subfolder+name+'_'+str(label)+'.png')

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
    try:
        img_id, img_pil = prepare_image(img_path,600,2800)
        save_img(img_pil, subfolder, img_id, "Bone")
        img_id, img_pil = prepare_image(img_path,80,200)
        save_img(img_pil, subfolder, img_id, "subdural")
        img_id, img_pil, = prepare_image(img_path,40,80)
        save_img(img_pil, subfolder, img_id, "brain")
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

    prepare_images_njobs(glob(Input), out_preprocessing_path)

def epochVal(model, dataLoader,pred_batch_size,c_pred):
    model.eval ()
    lossValNorm = 0
    valLoss = 0

    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    for i, input in enumerate (dataLoader):
        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_pred)/pred_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')
        feature = model.module.densenet121(inputs)
        feature = model.module.relu(feature)
        feature = model.module.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        for index, name in enumerate(names):
            if name not in features_list:
                features_list[name] = feature[index, :].cpu().detach().numpy() / 10
            else:
                features_list[name] += feature[index, :].cpu().detach().numpy() / 10
        feature = model.module.mlp(feature)
        feature = feature.sigmoid()

    valLoss = valLoss / lossValNorm

    auc = computeAUROC(outGT, outPRED, 3)
    auc = [round(x, 4) for x in auc]
    loss_list, loss_sum = weighted_log_loss(outPRED, outGT)

    return valLoss, auc, loss_list, loss_sum

def classification_net():
    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    df_all = pd.read_csv(csv_path)
    kfold_path_val = './output/test.txt'

    with open(snapshot_path + '/log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_fold])

    f_val = open(kfold_path_val, 'r')
    c_val = f_val.readlines()
    f_val.close()
    c_val = [s.replace('\n', '') for s in c_val]

    val_transform = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0,
                                 p=1.0)
    ])
    val_dataset = Dataset_val_by_study_context(df_all, c_val, val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False)

    model = eval(model_name + '()')
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    if path is not None:
        print(path)
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)["state_dict"])

    valLoss, auc, loss_list, loss_sum = epochVal(model, val_loader, loss_cls, c_val, val_batch_size)


if __name__ == '__main__':
    #STEP1 :Preprocessing
    Input_path = ""
    out_preprocessing_path = "" #make sure it end with /
    prepare_data(Input_path,out_preprocessing_path)
    #STEP2: Classification Network
    test_csv_path = ""
    Image_size = ""
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    batch_size = val_batch_size
    print(Image_size)
    print(train_batch_size)
    print(val_batch_size)
    model_snapshot_path = args.snapshot_path.replace('\n', '').replace('\r', '') + '/'
    kfold_path = '../data/fold_5_by_study_image/'
    df_test = pd.read_csv(test_csv_path)
    c_test = list(set(df_test['filename'].values.tolist()))
    df_all = pd.read_csv(csv_path)
    is_center = False
    is_aug = True
    num_aug = 10
    val_truth_oof = df_all.sort_values(by='filename').reset_index(drop=True).loc[:,
                    ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']].values

    out = 0 0 0 #Any ISk Hom

    #2,256,256
    #0 -> ISK
    #1 -> Hom
