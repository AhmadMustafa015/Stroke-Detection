import torch.utils.data as data
import torch
import albumentations
import cv2
import numpy as np
import random
import math
from keras.utils import to_categorical
import settings
from settings import png_out_path

def generate_transforms(image_size):
    IMAGENET_SIZE = image_size

    train_transform = albumentations.Compose([
        albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])

    val_transform = albumentations.Compose([
        albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])

    return train_transform, val_transform
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
                 transform=None
                 ):
        self.df = df  # [df['filename'].isin(name_list)]
        self.name_list = name_list
        self.transform = transform

    def __getitem__(self, idx):
        study_name = self.name_list[idx % len(self.name_list)]
        study_train_df = self.df[self.df['filename'] == int(study_name)]
        # study_index = random.choice(generate_random_list(study_train_df.shape[0]-1))

        # slice_id = study_name + '_' + str(study_index)
        filename = study_train_df['filename'].values
        if study_train_df['any'].values == 0:
            fullname = str(filename[0]) + '_0.png'
            assert False, "Error: Segmentation Training for stroke cases"
            #dicom_path + ISKEMI\MASK\ + str(filename[0]) + '.png'
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
        label = to_categorical(label, num_classes=2).astype(np.uint8)[1:] #Remove background #multi-class segmentation

        if random.random() < 0.5:
            image_cat = cv2.cvtColor(image_cat, cv2.COLOR_BGR2RGB)

        image_cat = aug_image(image_cat, is_infer=False)

        if self.transform is not None:
            augmented = self.transform(image=image_cat)
            image_cat = augmented['image'].transpose(2, 0, 1)

        # print(label)
        # exit(0)

        return image_cat, label

    def __len__(self):
        return len(self.name_list) * 4


class Dataset_val_by_study_context(data.Dataset):
    def __init__(self,
                 df=None,
                 name_list=None,
                 transform=None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform

    def __getitem__(self, idx):

        study_name = self.name_list[idx % len(self.name_list)]
        study_train_df = self.df[self.df['filename'] == int(study_name)]
        filename = study_train_df['filename'].values
        if study_train_df['any'].values == 0:
            fullname = str(filename[0]) + '_0.png'
        elif study_train_df['ISKEMI'].values == 1:
            fullname = str(filename[0]) + '_1.png'
        elif study_train_df['KANAMA'].values == 1:
            fullname = str(filename[0]) + '_2.png'
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
        label = torch.FloatTensor(study_train_df[study_train_df['filename'] == filename].loc[:, 'any':'KANAMA'].values)
        image_cat = aug_image(image_cat, is_infer=True)

        if self.transform is not None:
            augmented = self.transform(image=image_cat)
            image_cat = augmented['image'].transpose(2, 0, 1)

        return image_cat, label

    def __len__(self):
        return len(self.name_list)

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
        image[:,:,0:3] = np.rot90(image[:,:,0:3])
    return image

#===================================================origin=============================================================
def random_cropping(image, ratio=0.8, is_random = True):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    return zeros

def cropping(image, ratio=0.8, code = 0):
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
        return image

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
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
                img[x1:x1 + h, y1:y1 + w,:] = 0.0
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
    return image


def aug_image(image, is_infer=False):
    if is_infer:
        image = randomHorizontalFlip(image, u=0)
        image = np.asarray(image)
        image = cropping(image, ratio=0.8, code=0)
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

        ratio = random.uniform(0.6,0.99)
        image = random_cropping(image, ratio=ratio, is_random=True)
        return image



def generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):
    train_dataset = Dataset_train_by_study_context(df_all, c_train, train_transform)
    val_dataset = Dataset_val_by_study_context(df_all, c_val, val_transform)

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
