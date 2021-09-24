# ============ Basic imports ============#e
import os
import sys

sys.path.insert(0, '..')
import time
import gc
import pandas as pd
import cv2
import csv
from torch.utils.data import DataLoader
from dataset_loader import *
from tuils.tools import *
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
import math
import argparse
from net.models import *


def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
    return image


def randomVerticleFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
    return image


def randomRotate90(image, u=0.5):
    if np.random.random() < u:
        image[:, :, 0:3] = np.rot90(image[:, :, 0:3])
    return image


def random_cropping(image, ratio=0.8, is_random=True):
    height, width, _ = image.shape
    target_h = int(height * ratio)
    target_w = int(width * ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = (width - target_w) // 2
        start_y = (height - target_h) // 2

    zeros = image[start_y:start_y + target_h, start_x:start_x + target_w, :]
    zeros = cv2.resize(zeros, (width, height))
    return zeros


def cropping(image, ratio=0.8, code=0):
    height, width, _ = image.shape
    target_h = int(height * ratio)
    target_w = int(width * ratio)

    if code == 0:
        start_x = (width - target_w) // 2
        start_y = (height - target_h) // 2

    elif code == 1:
        start_x = 0
        start_y = 0

    elif code == 2:
        start_x = width - target_w
        start_y = 0

    elif code == 3:
        start_x = 0
        start_y = height - target_h

    elif code == 4:
        start_x = width - target_w
        start_y = height - target_h

    elif code == -1:
        return image

    zeros = image[start_y:start_y + target_h, start_x:start_x + target_w, :]
    zeros = cv2.resize(zeros, (width, height))
    return zeros


def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1 + h, y1:y1 + w, :] = 0.0
            else:
                print('!!!!!!!! random_erasing dim wrong!!!!!!!!!!!')
                return

            return img

    return img


def randomShiftScaleRotate(image,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
    return image


def aug_image(image, is_infer=False, augment=[0, 0]):
    if is_infer:
        image = randomHorizontalFlip(image, u=augment[0])
        image = np.asarray(image)
        image = cropping(image, ratio=0.8, code=augment[1])
        return image

    else:
        image = randomHorizontalFlip(image)
        height, width, _ = image.shape
        image = randomShiftScaleRotate(image,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-30, 30))

        image = cv2.resize(image, (width, height))
        image = random_erasing(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3)

        ratio = random.uniform(0.6, 0.99)
        image = random_cropping(image, ratio=ratio, is_random=True)

        return image


valid_transform_aug = albumentations.Compose([

    albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
])

valid_transform_pure = albumentations.Compose([

    albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
])


class PredictionDatasetPure:
    def __init__(self, name_list, mode):
        self.name_list = name_list
        self.mode = mode
        self.png_out_path = ""

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        filename = self.name_list[idx % len(self.name_list)]
        image = cv2.imread(self.png_out_path + 'extracted_png_brain/' + filename + '.png', 0)
        image = cv2.resize(image, (256, 256))
        image_up = cv2.imread(self.png_out_path + 'extracted_png_subdural/' + filename + '.png',
                              0)  # we use one window for now
        image_up = cv2.resize(image_up, (256, 256))
        image_down = cv2.imread(self.png_out_path + 'extracted_png_bone/' + filename + '.png', 0)
        image_down = cv2.resize(image_down, (256, 256))

        image_cat = np.concatenate(
            [image_up[:, :, np.newaxis], image[:, :, np.newaxis], image_down[:, :, np.newaxis]],
            2)
        label = torch.FloatTensor([0, 0, 0])

        image_cat = valid_transform_pure(image=image_cat)['image'].transpose(2, 0, 1)

        return filename, image_cat, label


class PredictionDatasetAug:
    def __init__(self, name_list, mode):
        self.name_list = name_list
        self.mode = mode
        self.png_out_path = "./prediction_output/"

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        filename = self.name_list[idx % len(self.name_list)]
        filename = filename.split('_')[0]
        image = cv2.imread(self.png_out_path + 'brain/' + filename + '_brain.png', 0)
        image = cv2.resize(image, (256, 256))
        image_up = cv2.imread(self.png_out_path + 'subdural/' + filename + '_subdural.png',
                              0)  # we use one window for now
        image_up = cv2.resize(image_up, (256, 256))
        image_down = cv2.imread(self.png_out_path + 'bone/' + filename + '_bone.png', 0)
        image_down = cv2.resize(image_down, (256, 256))

        image_cat = np.concatenate(
            [image_up[:, :, np.newaxis], image[:, :, np.newaxis], image_down[:, :, np.newaxis]],
            2)
        zero_np = np.zeros((2,256,256))
        label = torch.Tensor(zero_np)
        #label = torch.FloatTensor([0, 0, 0])

        if random.random() < 0.5:
            image_cat = cv2.cvtColor(image_cat, cv2.COLOR_BGR2RGB)
        else:
            image_cat

        #image_cat = randomHorizontalFlip(image_cat, u=0.5)
        height, width, _ = image_cat.shape
        #ratio = random.uniform(0.6, 0.99)
        #image_cat = random_cropping(image_cat, ratio=ratio, is_random=True)
        image_cat = valid_transform_aug(image=image_cat)['image'].transpose(2, 0, 1)

        return filename, image_cat, label


def predict(model, name_list, batch_size: int, aug=False, mode='val', fold=0):
    if aug:
        loader = DataLoader(
            dataset=PredictionDatasetAug(name_list, mode),
            shuffle=False,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True
        )
    else:
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
        labels = labels.contiguous().cuda(non_blocking=True)
        all_truth = torch.cat((all_truth, labels), 0)
        with torch.no_grad():
            output = model(inputs)
            print(output.shape)



        feature = torch.round(output) #change the threshold from here
        all_outputs = torch.cat((all_outputs, feature.data), 0)
        print(all_outputs.shape)
        all_names.extend(names)

    datanpGT = all_truth.cpu().numpy()
    datanpPRED = all_outputs.cpu().numpy()
    return datanpPRED, all_names, datanpGT


def group_aug(val_p_aug, val_names_aug, val_truth_aug):
    """
    Average augmented predictions
    :param val_p_aug:
    :return:
    """
    df_prob = pd.DataFrame(val_p_aug)
    df_prob['id'] = val_names_aug

    df_truth = pd.DataFrame(val_truth_aug)
    df_truth['id'] = val_names_aug

    g_prob = df_prob.groupby('id').mean()
    g_prob = g_prob.reset_index()
    g_prob = g_prob.sort_values(by='id')

    g_truth = df_truth.groupby('id').mean()
    g_truth = g_truth.reset_index()
    g_truth = g_truth.sort_values(by='id')

    return g_prob.drop('id', 1).values, g_truth['id'].values, g_truth.drop('id', 1).values


def predict_all(model_name, image_size):
    fold = 0
    print(fold)

    if not os.path.exists(model_snapshot_path + 'prediction/'):
        os.makedirs(model_snapshot_path + 'prediction/')
    if not os.path.exists(model_snapshot_path + 'prediction/npy_train/'):
        os.makedirs(model_snapshot_path + 'prediction/npy_train/')
    if not os.path.exists(model_snapshot_path + 'prediction/npy_test/'):
        os.makedirs(model_snapshot_path + 'prediction/npy_test/')

    prediction_path = model_snapshot_path + 'prediction/fold_{fold}'.format(fold=fold)

    model = eval(model_name + '()')
    model = nn.DataParallel(model).cuda()

    state = torch.load(model_snapshot_path + 'unet_best_weights.pth')

    epoch = state['epoch']
    best_valid_loss = state['valLoss']
    model.load_state_dict(state['state_dict'])
    print(epoch, best_valid_loss)

    model.eval()

    test_p_aug, test_names_aug, test_truth_aug = predict(model, name_list, batch_size, True, 'test', fold)

    All_final_labels = []
    for batch in range(0,test_p_aug.shape[0]):
        ISKEMIA = test_p_aug[batch,0,:,:]
        ISKEMIA = np.round(cv2.resize(ISKEMIA, (512, 512), cv2.INTER_CUBIC))
        BLEEDING = test_p_aug[batch, 1,:,:]
        BLEEDING = np.round(cv2.resize(BLEEDING, (512, 512), cv2.INTER_CUBIC)) * 2
        final_label = ISKEMIA + BLEEDING
        final_label[final_label > 2] = 2
        All_final_labels.append([test_names_aug[batch], final_label])
        if not os.path.exists("./prediction_segmentation/"):
            os.makedirs("./prediction_segmentation/")
        if not os.path.exists("./prediction_segmentation/submission/"):
            os.makedirs("./prediction_segmentation/submission/")
        if not os.path.exists("./prediction_segmentation/visualize/"):
            os.makedirs("./prediction_segmentation/visualize/")

        cv2.imwrite("./prediction_segmentation/submission/" + test_names_aug[batch] + ".png", final_label)
        cv2.imwrite("./prediction_segmentation/visualize/" + test_names_aug[batch] + ".png", final_label * 125)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='unet_brain_seg', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=1024, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=4, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=4, help='val_batch_size')

    parser.add_argument("-spth", "--snapshot_path", type=str,
                        default='unet_brain_seg', help='epoch')

    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    batch_size = val_batch_size
    workers = 6
    print(Image_size)
    print(train_batch_size)
    print(val_batch_size)
    name_list = []
    out_preprocessing_path = "./prediction_output/brain/"
    for (dirpath, dirnames, filenames) in os.walk(out_preprocessing_path):
        name_list.extend(filenames)
        break

    model_snapshot_path = './data_test/'
    kfold_path = './data_test/unet_brain_seg/model_epoch_109_0.pth'

    backbone = args.backbone
    print(backbone)
    predict_all(backbone, Image_size)


