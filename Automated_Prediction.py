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
    print(img_path)
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

    prepare_images_njobs(glob(Input + '/*'), out_preprocessing_path)
"""
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

valid_transform_pure = albumentations.Compose([

    albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
])
class PredictionDatasetClass:
    def __init__(self,
                 name_list=None,
                 transform=None,
                 png_out_path=None,
                 png_input_path = None
                 ):
        self.name_list = name_list
        self.transform = transform
        self.png_out_path = png_out_path

    def __getitem__(self, idx):

        study_name = self.name_list[idx % len(self.name_list)]


        image = cv2.imread(self.png_out_path + 'extracted_png_brain/' + study_name + '.png', 0)
        image = cv2.resize(image, (512, 512))
        image_up = cv2.imread(self.png_out_path + 'extracted_png_subdural/' + study_name + '.png', 0)  # we use one window for now
        image_up = cv2.resize(image_up, (512, 512))
        image_down = cv2.imread(self.png_out_path + 'extracted_png_bone/' + study_name + '.png', 0)
        image_down = cv2.resize(image_down, (512, 512))

        image_cat = np.concatenate([image_up[:, :, np.newaxis], image[:, :, np.newaxis], image_down[:, :, np.newaxis]],
                                   2)
        #label = torch.FloatTensor(study_train_df[study_train_df['filename'] == filename].loc[:, 'any':'KANAMA'].values)
        #image_cat = aug_image(image_cat, is_infer=True)
        label = torch.FloatTensor([0, 0, 0])
        image_cat = valid_transform_pure(image=image_cat)['image'].transpose(2, 0, 1)

        return study_name, image_cat, label

    def __len__(self):
        return len(self.name_list)

def predict_classification():
    loader = DataLoader(
        dataset=PredictionDatasetPure(name_list, mode),
        shuffle=False,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True
    )
    model.eval()

    all_names = []
    all_outputs = torch.FloatTensor().cuda()
    all_truth = torch.FloatTensor().cuda()
    features_list = {}
    for names, inputs, labels in tqdm(loader, desc='Predict'):
        labels = labels.view(-1, 3).contiguous().cuda(async=True)
        all_truth = torch.cat((all_truth, labels), 0)
        with torch.no_grad():
            inputs = torch.autograd.variable(inputs).cuda(async=True)
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
        all_outputs = torch.cat((all_outputs, feature.data), 0)
        all_names.extend(names)
    datanpGT = all_truth.cpu().numpy()
    datanpPRED = all_outputs.cpu().numpy()
    return datanpPRED, all_names, datanpGT
    """
if __name__ == '__main__':
    #STEP1 :Preprocessing
    Input_path = "./Testdata"
    out_preprocessing_path = "./prediction_output/" #make sure it end with /
    prepare_data(Input_path,out_preprocessing_path)
    """
    #Extract the test from csv to folder
    Kfold_path_val = "test.txt"
    f_val = open(kfold_path_val, 'r')
    c_val = f_val.readlines()
    f_val.close()
    c_val = [s.replace('\n', '') for s in c_val]
    for path in glob.glob(dir + '/*/*.dcm'):
        #do stuff
"""


"""
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
"""