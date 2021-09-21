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


class PredictionDatasetClass:
    def __init__(self, name_list, df_train, df_test, n_test_aug, mode):
        self.name_list = name_list
        if mode == 'val':
            self.df = df_train[df_train['filename'].isin(name_list)]
        elif mode == 'test':
            self.df = df_test[df_test['filename'].isin(name_list)]
        self.n_test_aug = n_test_aug
        self.mode = mode

    def __len__(self):
        return len(self.name_list) * self.n_test_aug

    def __getitem__(self, idx):
        if self.mode == 'val':
            filename = self.name_list[idx % len(self.name_list)]
            image_cat = cv2.imread('/home1/kaggle_rsna2019/process/train_concat_3images_256/' + filename)
            label = torch.FloatTensor(self.df[self.df['filename'] == filename].loc[:, 'any':'subdural'].values)

        if self.mode == 'test':
            filename = self.name_list[idx % len(self.name_list)]
            image_cat = cv2.imread('/home1/kaggle_rsna2019/process/stage2_test_concat_3images/' + filename)
            image_cat = cv2.resize(image_cat, (256, 256))
            label = torch.FloatTensor([0, 0, 0])

        image_cat = aug_image(image_cat, is_infer=True)
        image_cat = valid_transform_pure(image=image_cat)['image'].transpose(2, 0, 1)

        return filename, image_cat, label

def predict_classification():
    loader = DataLoader(
        dataset=PredictionDatasetPure(name_list, df_all, df_test, n_test_aug, mode),
        shuffle=False,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True
    )

if __name__ == '__main__':
    #STEP1 :Preprocessing
    Input_path = ""
    out_preprocessing_path = "" #make sure it end with /
    prepare_data(Input_path,out_preprocessing_path)
    #STEP2: Classification Network
    model_name = "Dense"
    model_snapshot_path = "./data_test/" + model_name + "/"
    model = eval(model_name + '()')
    model = nn.DataParallel(model).cuda()
    state = torch.load(model_snapshot_path + 'model_epoch_best.pth')

    epoch = state['epoch']
    best_valid_loss = state['valLoss']
    model.load_state_dict(state['state_dict'])
    print(epoch, best_valid_loss)

    model.eval()










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


def predict_segmentation():
    input_image = Image.open(filename)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    print(torch.round(output[0]))
    print(torch.round(output[1]))