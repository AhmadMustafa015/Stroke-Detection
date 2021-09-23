import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data

import torch.utils.data as data
from tqdm import tqdm
from net.models import *
from data_loader_seg import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW, RAdam
from tuils.loss_function import *
import torch.nn.functional as F
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(1992)
torch.cuda.manual_seed(1992)
np.random.seed(1992)
random.seed(1992)
from PIL import ImageFile
import sklearn
import copy
torch.backends.cudnn.benchmark = True
import argparse
from tuils.tools import DiceLoss

def epochVal(model, dataLoader, loss_cls, c_val, val_batch_size):
    model.eval ()
    lossValNorm = 0
    valLoss = 0

    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    for i, (input, target) in enumerate (dataLoader):
        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_val)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')
        target = target.contiguous().cuda()
        outGT = torch.cat((outGT, target), 0)
        varInput = torch.autograd.Variable(input)
        varTarget = torch.autograd.Variable(target.contiguous().cuda())
        varOutput = model(varInput)
        lossvalue = loss_cls(varOutput, varTarget)
        valLoss = valLoss + lossvalue.item()
        varOutput = varOutput.sigmoid()

        outPRED = torch.cat((outPRED, varOutput.data), 0)
        lossValNorm += 1

    valLoss = valLoss / lossValNorm

    #auc = computeAUROC(outGT, outPRED, 3)
    #auc = [round(x, 4) for x in auc]
    auc = 0
    #loss_list, loss_sum = weighted_log_loss(outPRED, outGT)
    loss_list, loss_sum = 1,1
    return valLoss, auc, loss_list, loss_sum

def train(model_name,image_size):
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss']
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    df_all = pd.read_csv(csv_path)
    df_all_filtered = df_all[df_all['any'] == 0]
    df_all = df_all[df_all['any'] == 1]
    kfold_path_train = 'output/fold_5_by_study/'
    kfold_path_val = 'output/fold_5_by_study/'
    Kfold_path_test = "output/test.txt"
    Kfold_path_train_bat = "output/train_baturalp.txt"
    num_fold = 0 #for now
    with open(snapshot_path + '/log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_fold])

    f_train = open(kfold_path_train + 'fold' + str(num_fold) + '/train.txt', 'r')
    f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r')
    f_test = open(Kfold_path_test, 'r')
    f_train_bat = open(Kfold_path_train_bat, 'r')

    c_train = f_train.readlines()
    c_val = f_val.readlines()
    f_train.close()
    f_val.close()
    c_train = [s.replace('\n', '') for s in c_train]
    c_val = [s.replace('\n', '') for s in c_val]
    c_test = f_test.readlines()
    f_test.close()
    c_test = [s.replace('\n', '') for s in c_test]
    c_bat = f_train_bat.readlines()
    f_train_bat.close()
    c_bat = [s.replace('\n', '') for s in c_bat]
    c_train = c_train + c_test
    for index, row in df_all_filtered.iterrows():
        if str(row['filename']) in c_train:
            c_train.remove(str(row['filename']))
        if str(row['filename']) in c_val:
            c_val.remove(str(row['filename']))
    c_train += c_bat

    # for debug
    # c_train = c_train[0:1000]
    # c_val = c_val[0:4000]

    print('train dataset study num:', len(c_train), '  val dataset image num:', len(c_val))
    with open(snapshot_path + '/log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['train dataset:', len(c_train), '  val dataset:', len(c_val)])
        writer.writerow(['train_batch_size:', train_batch_size, 'val_batch_size:', val_batch_size])

    train_transform, val_transform, train_mask_transform, val_mask_transform = generate_transforms(image_size)
    train_loader, val_loader = generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val,
                                                       val_transform, val_batch_size, workers,train_mask_transform, val_mask_transform)

    loaders = {"train": train_loader, "valid": val_loader}

    model = eval(model_name + '()')
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
    scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
    model = torch.nn.DataParallel(model)
    loss_cls = DiceLoss().cuda()

    trMaxEpoch = 200
    for epochID in tqdm(range(0,trMaxEpoch), total=trMaxEpoch):
        epochID = epochID + 0

        start_time = time.time()
        model.train()
        trainLoss = 0
        lossTrainNorm = 10

        if epochID < 10:
            pass
        elif epochID < 80:
            if epochID != 10:
                scheduler.step()
                scheduler = warm_restart(scheduler, T_mult=2)
        else:
            optimizer.param_groups[0]['lr'] = 1e-5

        for batchID, (input, target) in enumerate(train_loader):
            if batchID == 0:
                ss_time = time.time()

            print(str(batchID) + '/' + str(int(len(c_train) / train_batch_size)) + '     ' + str(
                (time.time() - ss_time) / (batchID + 1)), end='\r')
            varInput = torch.autograd.Variable(input)
            target = target.contiguous().cuda()  # change to 3 in ourcase
            varTarget = torch.autograd.Variable(target.contiguous().cuda())
            varOutput = model(varInput)
            lossvalue = loss_cls(varOutput, varTarget)
            trainLoss = trainLoss + lossvalue.item()
            lossTrainNorm = lossTrainNorm + 1

            lossvalue.backward()
            optimizer.step()
            optimizer.zero_grad()
            del lossvalue

        trainLoss = trainLoss / lossTrainNorm

        if (epochID + 1) % 5 == 0 or epochID > 79 or epochID == 0:
            valLoss, _, _, _ = epochVal(model, val_loader, loss_cls, c_val, val_batch_size)

        epoch_time = time.time() - start_time

        if (epochID + 1) % 5 == 0 or epochID > 79:
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'valLoss': valLoss},
                       snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth')

        result = [epochID,
                  round(optimizer.state_dict()['param_groups'][0]['lr'], 6),
                  round(epoch_time, 0),
                  round(trainLoss, 5),
                  round(valLoss, 5),
                  #'auc:', auc,
                  #'loss:', loss_list,
                  ]#loss_sum]

        print(result)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result)

    del model

if __name__ == '__main__':
    csv_path = 'output/train.csv'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='unet_brain_seg', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=256, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=64, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=64, help='val_batch_size')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='unet_brain_seg', help='epoch')
    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = 24
    backbone = args.backbone
    print(backbone)
    print('image size:', Image_size)
    print('train batch size:', train_batch_size)
    print('val batch size:', val_batch_size)
    snapshot_path = 'data_test/' + args.model_save_path.replace('\n', '').replace('\r', '')
    train(backbone, Image_size)