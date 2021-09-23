import torch.utils.data as data
import torch
import albumentations
import cv2
import numpy as np
import random
import math
from tensorflow.keras.utils import to_categorical
import settings
from settings import png_out_path

def generate_transforms(image_size):
    IMAGENET_SIZE = image_size

    train_transform = albumentations.Compose([
        albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])
    train_mask_transform = albumentations.Compose([
        albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE)
    ])

    val_transform = albumentations.Compose([
        albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])
    val_mask_transform = albumentations.Compose([
        albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE)
    ])
    return train_transform, val_transform, train_mask_transform, val_mask_transform
def generate_random_list(length):
    new_list = []

    for i in range(length):
        if i <= length/2:
            weight = int(i/4)
        else:
            weight = int((length - i)/4)
        weight = np.max([1, weight])
        new_list += [i]*weight

    return new_list


class Dataset_train_by_study_context(data.Dataset):
    def __init__(self,
                 df=None,
                 name_list=None,
                 transform=None,
                 mask_trans=None
                 ):
        self.df = df  # [df['filename'].isin(name_list)]
        self.name_list = name_list
        self.transform = transform
        self.mask_trans = mask_trans
    def __getitem__(self, idx):
        study_name = self.name_list[idx % len(self.name_list)]
        is_bat = study_name.split('@')[0]
        if is_bat != 'baturalp':
            study_train_df = self.df[self.df['filename'] == int(study_name)]
            filename = study_train_df['filename'].values
            if study_train_df['any'].values == 0:
                fullname = str(filename[0]) + '_0.png'
                assert False, "Error: Segmentation Training for stroke cases"
                # dicom_path + ISKEMI\MASK\ + str(filename[0]) + '.png'
            elif study_train_df['ISKEMI'].values == 1:
                fullname = str(filename[0]) + '_1.png'
                label = np.array(cv2.imread(settings.dicom_path + 'ISKEMI/MASK/' + str(filename[0]) + '.png', 0))
            elif study_train_df['KANAMA'].values == 1:
                fullname = str(filename[0]) + '_2.png'
                label = np.array(cv2.imread(settings.dicom_path + 'KANAMA/MASK/' + str(filename[0]) + '.png', 0))
            else:
                print("Error wrong image label")
        else:
            fullname = study_name
            label = np.array(cv2.imread('./Baturalp_labels/Selected MASK/' + str(fullname), 0))
        image = cv2.imread(png_out_path + 'extracted_png_brain/' + fullname, 0)
        image = cv2.resize(image, (512, 512))
        image_up = cv2.imread(png_out_path + 'extracted_png_subdural/' + fullname, 0)  # we use one window for now
        image_up = cv2.resize(image_up, (512, 512))
        image_down = cv2.imread(png_out_path + 'extracted_png_bone/' + fullname, 0)
        image_down = cv2.resize(image_down, (512, 512))

        image_cat = np.concatenate([image_up[:, :, np.newaxis], image[:, :, np.newaxis], image_down[:, :, np.newaxis]],
                                   2)
        label = to_categorical(label, num_classes=3).astype(np.uint8) #multi-class segmentation
        label = np.moveaxis(np.array(label), 2, 0)[1:]  #Remove background
        if random.random() < 0.5:
            image_cat = cv2.cvtColor(image_cat, cv2.COLOR_BGR2RGB)

        image_cat,label = aug_image(image_cat,label, is_infer=False)

        if self.transform is not None:
            augmented = self.transform(image=image_cat)
            image_cat = augmented['image'].transpose(2, 0, 1) #same as np.moveaxis(a,-1,0)
            label_aug0 = self.mask_trans(image=label[0])
            label_aug1 = self.mask_trans(image=label[1])
            label_10 = []
            label_10.append(label_aug0['image'])
            label_10.append(label_aug1['image'])
            label = np.array(label_10)
        # print(label)
        # exit(0)
        #mask = mask.transpose(2, 0, 1)

        #image_tensor = torch.from_numpy(image.astype(np.float32))
        #mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return image_cat, label

    def __len__(self):
        return len(self.name_list) * 4


class Dataset_val_by_study_context(data.Dataset):
    def __init__(self,
                 df=None,
                 name_list=None,
                 transform=None,
                 mask_trans=None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform
        self.mask_trans = mask_trans

    def __getitem__(self, idx):

        study_name = self.name_list[idx % len(self.name_list)]
        study_train_df = self.df[self.df['filename'] == int(study_name)]
        filename = study_train_df['filename'].values
        if study_train_df['any'].values == 0:
            fullname = str(filename[0]) + '_0.png'
            assert False, "Error: Segmentation Training for stroke cases"
            # dicom_path + ISKEMI\MASK\ + str(filename[0]) + '.png'
        elif study_train_df['ISKEMI'].values == 1:
            fullname = str(filename[0]) + '_1.png'
            label = np.array(cv2.imread(settings.dicom_path + 'ISKEMI/MASK/' + str(filename[0]) + '.png', 0))
        elif study_train_df['KANAMA'].values == 1:
            fullname = str(filename[0]) + '_2.png'
            label = np.array(cv2.imread(settings.dicom_path + 'KANAMA/MASK/' + str(filename[0]) + '.png', 0))
        else:
            print("Error wrong image label")
        image = cv2.imread(png_out_path + 'extracted_png_brain/' + fullname, 0)
        image = cv2.resize(image, (512, 512))
        image_up = cv2.imread(png_out_path + 'extracted_png_subdural/' + fullname, 0)  # we use one window for now
        image_up = cv2.resize(image_up, (512, 512))
        image_down = cv2.imread(png_out_path + 'extracted_png_bone/' + fullname, 0)
        image_down = cv2.resize(image_down, (512, 512))

        image_cat = np.concatenate([image_up[:, :, np.newaxis], image[:, :, np.newaxis], image_down[:, :, np.newaxis]],
                                   2)
        label = to_categorical(label, num_classes=3).astype(np.uint8)  #multi-class segmentation
        label = np.moveaxis(np.array(label), 2, 0)[1:]  #Remove background
        image_cat,label = aug_image(image_cat,label, is_infer=True)

        if self.transform is not None:
            augmented = self.transform(image=image_cat)
            image_cat = augmented['image'].transpose(2, 0, 1)
            label_aug0 = self.mask_trans(image=label[0])
            label_aug1 = self.mask_trans(image=label[1])
            label_10 = []
            label_10.append(label_aug0['image'])
            label_10.append(label_aug1['image'])
            label = np.array(label_10)

        return image_cat, label

    def __len__(self):
        return len(self.name_list)

def randomHorizontalFlip(image,label, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        label[0] = cv2.flip(label[0], 1)
        label[1] = cv2.flip(label[1], 1)
    return image,label

def randomVerticleFlip(image,label, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        label[0] = cv2.flip(label[0], 0)
        label[1] = cv2.flip(label[1], 0)
    return image,label

def randomRotate90(image,label, u=0.5):
    if np.random.random() < u:
        image[:,:,0:3] = np.rot90(image[:,:,0:3])
        label[0:2,:, :] = np.rot90(label[0:2, :, :])
    return image,label

#===================================================origin=============================================================
def random_cropping(image, label, ratio=0.8, is_random = True):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2
    zer_lab = []
    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    zer_lab.append(label[0,start_y:start_y + target_h, start_x:start_x + target_w])
    zer_lab[0] = cv2.resize(zer_lab[0], (width, height))
    zer_lab.append(label[1,start_y:start_y + target_h, start_x:start_x + target_w])
    zer_lab[1] = cv2.resize(zer_lab[1], (width, height))
    zer_lab = np.array(zer_lab)
    return zeros,zer_lab

def cropping(image, label, ratio=0.8, code = 0):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if code==0:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

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
        return image,label

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    zer_lab = []
    zer_lab.append(label[0, start_y:start_y + target_h, start_x:start_x + target_w])
    zer_lab[0] = cv2.resize(zer_lab[0], (width, height))
    zer_lab.append(label[1, start_y:start_y + target_h, start_x:start_x + target_w])
    zer_lab[1] = cv2.resize(zer_lab[1], (width, height))
    zer_lab = np.array(zer_lab)
    return zeros,zer_lab

def random_erasing(img,label, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    if random.uniform(0, 1) > probability:
        return img,label

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
                img[x1:x1 + h, y1:y1 + w,:] = 0.0
                label[0,x1:x1 + h, y1:y1 + w] = 0.0
                label[1,x1:x1 + h, y1:y1 + w] = 0.0
            else:
                print('!!!!!!!! random_erasing dim wrong!!!!!!!!!!!')
                return

            return img,label
    return img,label

def randomShiftScaleRotate(image, label,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):

    if np.random.random() < u:
        height, width, channels = image.shape

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
        label_conc = []
        label_conc.append(cv2.warpPerspective(label[0], mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,)))
        label_conc.append(cv2.warpPerspective(label[1], mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,)))
        label = np.array(label_conc)
    return image,label


def aug_image(image,label, is_infer=False):
    if is_infer:
        image,label = randomHorizontalFlip(image,label, u=0)
        image = np.asarray(image)
        label = np.asarray(label)
        image,label = cropping(image,label, ratio=0.8, code=0)
        return image,label

    else:
        image,label = randomHorizontalFlip(image,label)
        height, width, _ = image.shape
        image,label = randomShiftScaleRotate(image,label,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-30, 30))

        image = cv2.resize(image, (width, height))
        label_conc = []
        label_conc.append(cv2.resize(label[0], (width, height)))
        label_conc.append(cv2.resize(label[1], (width, height)))
        label = np.array(label_conc)
        image,label = random_erasing(image,label, probability=0.5, sl=0.02, sh=0.4, r1=0.3)

        ratio = random.uniform(0.6,0.99)
        image,label = random_cropping(image, label, ratio=ratio, is_random=True)
        return image,label



def generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers,train_mask_transform, val_mask_transform):
    train_dataset = Dataset_train_by_study_context(df_all, c_train, train_transform, train_mask_transform)
    val_dataset = Dataset_val_by_study_context(df_all, c_val, val_transform, val_mask_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, val_loader
