import copy
import logging
import math
from os.path import join as pjoin
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torchvision import transforms
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import torch.utils.data as data
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import time
import sys
from datetime import datetime
import argparse


import lossfun

import utils
from dataset import MyDataset
from utils import *
from models.mamveseg import MamVeSeg



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
HISTOLOGY_DATA_PATH = './data/vessels'  # Containing two folders named train and test, and in each subforlder contains two folders named data and label.
RADIOLOGY_DATA_PATH = './data/private'  # Containing two folders named train and test, and in each subforlder contains two folders named data and label.
num_classes = 2

IMG_SIZE = 512


def main():
    '''
    model_type:  default: MamVeSeg
    alpha: ratio of the loss of nuclei mask loss, dafault=0.3
    beta: ratio of the loss of normal edge segmentation, dafault=0.35
    gamma: ratio of the loss of cluster edge segmentation, dafault=0.35
    sharing_ratio: ratio of sharing proportion of decoders, default=0.5
    random_seed: set the random seed for splitting dataset
    dataset: Radiology(grayscale) or Histology(rgb), default=Histology
    num_epoch: number of epoches
    lr: learning rate
    model_path: if used pretrained model, put the path to the pretrained model here
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, default="MamVeSeg",
                        help="declare the model type to use, currently only support input being MamVeSeg")
    parser.add_argument("--alpha", required=True, default=0.3, help="coeffiecient of the weight of nuclei mask loss")
    parser.add_argument("--beta", required=True, default=0.35, help="coeffiecient of the weight of normal edge loss")
    parser.add_argument("--gamma", required=True, default=0.35, help="coeffiecient of the weight of cluster edge loss")
    parser.add_argument("--sharing_ratio", required=True, default=0.5, help=" ratio of sharing proportion of decoders")
    parser.add_argument("--random_seed", required=True, help="random seed")
    parser.add_argument("--batch_size", required=True, help="batch size")
    parser.add_argument("--dataset", required=True, default="STARE", help="Histology, Radiology")
    parser.add_argument("--num_epoch", required=True, help='number of epoches')
    parser.add_argument("--lr", required=True, help="learning rate")
    parser.add_argument("--model_path", default=None, help="the path to the pretrained model")

    args = parser.parse_args()

    model_type = args.model_type
    dataset = args.dataset

    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)
    sharing_ratio = float(args.sharing_ratio)
    batch_size = int(args.batch_size)
    random_seed = int(args.random_seed)
    num_epoch = int(args.num_epoch)
    base_lr = float(args.lr)
    import warnings
    warnings.filterwarnings("ignore")
    if dataset == "STARE":
        channel = 3
        IMG_SIZE = 512
    elif dataset == "vessels":
        channel = 3
        IMG_SIZE = 512

    else:
        print("Wrong Dataset type")
        return 0

    model = MamVeSeg(img_size=IMG_SIZE, in_chans=channel)
    if args.model_path is not None:
        try:
            model.load_state_dict(torch.load(args.model_path))
        except Exception as err:
            print("{} In Loading previous model weights".format(err))
    # model.load_state_dict('')
    model.to(device)

    now = datetime.now()
    create_dir('./log')
    logging.basicConfig(filename='./log/log_{}_{}.txt'.format(model_type, dataset, str(now)), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d]', datefmt='%H%M%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(
        "Batch size : {} , epoch num: {}, alph: {}, beta : {}, gamma: {}, sharing_ratio = {}".format(batch_size,
                                                                                                     num_epoch, alpha,
                                                                                                     beta, gamma,
                                                                                                     sharing_ratio))
    if dataset == "private":
        # total_data = MyDataset()
        data_path = RADIOLOGY_DATA_PATH
        total_data = MyDataset(dir_path=data_path)
        train_set_size = int(len(total_data) * 0.8)
        test_set_size = len(total_data) - train_set_size

        train_set, test_set = data.random_split(total_data, [train_set_size, test_set_size],
                                                generator=torch.Generator().manual_seed(random_seed))
    elif dataset == "vessels":
        train_set = MyDataset(dir_path="./data/vessels/train",transform=None)
        test_set = MyDataset(dir_path="./data/vessels/test",transform=None)

        train_set_size = int(len(train_set))
        test_set_size = int(len(test_set))

        logging.info("train size {} test size {}".format(train_set_size, test_set_size))

    else:
        logging.info("Wrong Dataset type")
        return 0


    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    dataloaders = {"train": trainloader, "test": testloader}
    dataset_sizes = {"train": len(trainloader), "test": len(testloader)}
    logging.info("size train : {}, size test {} ".format(dataset_sizes["train"], dataset_sizes["test"]))


    test_loss = []
    train_loss = []
    Acc_list = []
    F1_list = []
    IOU_list = []
    SE_list = []
    SP_list = []
    AUC_list = []

    ce_loss1 = CrossEntropyLoss()
    dice_loss1 = DiceLoss(num_classes)
    ce_loss2 = CrossEntropyLoss()
    dice_loss2 = DiceLoss(num_classes)
    ce_loss3 = CrossEntropyLoss()
    dice_loss3 = DiceLoss(num_classes)
    dice_loss_dis = DiceLoss(num_classes)

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    best_loss = 100
    best_epoch = 0


    for epoch in range(num_epoch):
        # early stop, if the loss does not decrease for 50 epochs
        # if epoch > best_epoch + 50:
        #     break
        for phase in ['train', 'test']:
            running_loss = 0
            running_loss_wo_dis = 0
            running_loss_seg = 0
            s = time.time()  # start time for this epoch
            if phase == 'train':
                model.train()

                # Set model to training mode
            else:
                model.eval()
            metric_list = 0.0
            # aji_score_list = 0.0
            IoU_list = 0.0
            F1_score_list = 0.0
            se_list = 0.0
            sp_list = 0.0
            acc_list = 0.0
            auc_list = 0.0

            for i, d in enumerate(dataloaders[phase]):
                # if i == 1:
                #     break
                img, instance_seg_mask, semantic_seg_mask, normal_edge_mask, cluster_edge_mask = d
                label = semantic_seg_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype('int')
                # plt.imshow(semantic_seg_mask, cmap='gray')  # Display the ndarray as a grayscale image
                # plt.axis('off')  # Turn off axis
                # plt.show()
                # img = img.squeeze(0).squeeze(0).detach().cpu().numpy().astype('int')
                # img = np.transpose(img, (1, 2, 0))
                # plt.imshow(img)  # Display the ndarray as a grayscale image
                # plt.axis('off')  # Turn off axis
                # plt.show()
                img = img.float()
                img = img.to(device)

                instance_seg_mask = instance_seg_mask.to(device)
                semantic_seg_mask = semantic_seg_mask.to(device)



                # boundary_gt = (semantic_seg_mask != 0).long().cuda()
                # cluster_gt = (semantic_seg_mask == 2).long().cuda()
                # semantic_seg_mask2 = semantic_seg_mask.cpu().detach().numpy()
                # normal_edge_mask2 = normal_edge_mask.cpu().detach().numpy()
                # cluster_edge_mask2 = cluster_edge_mask.cpu().detach().numpy()

                cluster_edge_mask = cluster_edge_mask.to(device)

                # print('img shape ',img.shape)
                # print('semantic_seg_mask shape ',semantic_seg_mask.shape)

                output1, output2, output3 = model(img)

                # aff = fusion.iAFF()
                # aff = aff.cuda()
                # output1 = aff(output1,output2)




                if phase == 'test':
                    acoutput1 = output1.squeeze(0).detach().cpu().numpy()#2,512,512
                    prediction = np.argmax(acoutput1, 0)#512,512
                    # prediction = acoutput1[1]#512,512
                    #
                    # plt.imshow(prediction, cmap='gray')  # Display the ndarray as a grayscale image
                    # plt.axis('off')  # Turn off axis
                    # plt.show()

                    bout = output2.squeeze(0).detach().cpu().numpy()
                    bout= np.argmax(bout, 0)
                    cout = output3.squeeze(0).detach().cpu().numpy()
                    cout = np.argmax(cout, 0)


                    # metric_i = [calculate_acc(prediction, label), calculate_IoU(prediction, label), calculate_F1_score(prediction, label)]
                    # metric_list += np.array(metric_i)

                    # # 对 prediction 和 label 进行裁剪操作，保留第一个维度之外的部分
                    # prediction = prediction[0, :, :]
                    # label = label[0, :, :]

                    # print(prediction.shape)
                    # print(label.shape)

                    IoU = calculate_IoU(prediction, label)
                    F1_score = calculate_F1_score(prediction, label)
                    acc = calculate_acc(prediction, label)
                    se = compute_class_sens(prediction, label)
                    sp = compute_class_spec(prediction, label)
                    auc = calculate_auc (label.flatten(), prediction.flatten())
                    # hello world
                    acc_list += acc
                    IoU_list += IoU
                    F1_score_list += F1_score
                    se_list += se
                    sp_list += sp
                    auc_list = auc + auc_list



                loss_v = lossfun.soft_dice_cldice()

                semantic_seg_mask_v = semantic_seg_mask
                semantic_seg_mask_v = semantic_seg_mask_v.unsqueeze(1)

                normal_edge_mask_v=  normal_edge_mask
                normal_edge_mask_v =  normal_edge_mask_v.unsqueeze(1)

                cluster_edge_mask_v = cluster_edge_mask
                cluster_edge_mask_v = cluster_edge_mask_v.unsqueeze(1)

                loss_seg =  0.4 * ce_loss1(output1, semantic_seg_mask.long()) + 0.6 * dice_loss1(output1,semantic_seg_mask.float(),softmax=True)
                # loss_seg = loss_v(semantic_seg_mask_v.cuda(), output1)

                # loss_nor = 0.4 * ce_loss2(output2, normal_edge_mask.long()) + 0.6 * dice_loss2(output2,normal_edge_mask.float(),softmax=True)
                loss_nor = loss_v(normal_edge_mask_v.cuda(),output2)
                # loss_nor=0
                loss_clu = 0.4 * ce_loss3(output3, cluster_edge_mask.long())+ 0.6 * dice_loss3(output3,cluster_edge_mask.float(),softmax=True)

                if epoch < 10:
                    ratio_d = 1
                elif epoch < 20 and epoch > 10:
                    ratio_d = 0.7
                    base_lr = base_lr / 10
                elif epoch < 30 and epoch > 20:
                    ratio_d = 0.4
                    base_lr = base_lr / 10



                ### calculating the distillation loss
                m = torch.softmax(output1, dim=1)
                m = torch.argmax(m, dim=1)
                # m = m.squeeze(0)
                m = m.cpu().detach().numpy()


                c = torch.argmax(torch.softmax(output3, dim=1), dim=1)
                pred_edge_1 = edge_detection(m.copy(), channel)
                pred_edge_1 = torch.tensor(pred_edge_1).to(device)
                pred_edge_2 = output2 - output3
                pred_edge_2[pred_edge_2 < 0] = 0

                dis_loss = dice_loss_dis(pred_edge_2, pred_edge_1.float())

                ### calculating total loss
                loss =  alpha * loss_seg + beta * loss_nor + gamma * loss_clu + ratio_d * dis_loss

                running_loss += loss.item()
                running_loss_wo_dis += (
                            alpha * loss_seg + beta * loss_nor + gamma * loss_clu).item()  ## Loss without distillation loss
                running_loss_seg += loss_seg.item()  ## Loss for nuclei segmantation
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            metric_list = metric_list / dataset_sizes[phase]
            e = time.time()
            # aji_score_list = aji_score_list / dataset_sizes["test"]
            acc_list = acc_list / dataset_sizes["test"]
            sp_list = sp_list / dataset_sizes["test"]
            se_list = se_list / dataset_sizes["test"]
            IoU_list = IoU_list / dataset_sizes["test"]
            F1_score_list = F1_score_list / dataset_sizes["test"]
            auc_list = auc_list / dataset_sizes["test"]


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_wo_dis = running_loss_wo_dis / dataset_sizes[phase]  ## Epoch Loss without distillation loss
            epoch_loss_seg = running_loss_seg / dataset_sizes[phase]  ## Epoch Loss for nuclei segmantation
            logging.info('Epoch {},: loss {}, {},time {}'.format(epoch + 1, epoch_loss, phase, e - s))
            logging.info(
                'Epoch {},: loss without distillation {}, {},time {}'.format(epoch + 1, epoch_loss_wo_dis, phase,
                                                                             e - s))
            logging.info('Epoch {},: loss seg {}, {},time {}'.format(epoch + 1, epoch_loss_seg, phase, e - s))
            # logging.info(
            #     'Mean accuracy: {:.5f}, mean IoU: {:.5f}, mean F1 score: {:.5f}'.format(metric_list[0], metric_list[1],
            #                                                                             metric_list[2]))

            # logging.info('Epoch {},: aji_score {}, {},time {}'.format(epoch+1,  aji_score_list,phase,e-s))

            if phase == 'train':
                train_loss.append(epoch_loss)
                # train_loss_clus.append(loss_clu)
                # train_loss_nors.append(loss_nor)
                # train_loss_segs.append(loss_seg)

            else:
                test_loss.append(epoch_loss)
                Acc_list.append(acc_list)
                F1_list.append(F1_score_list)
                IOU_list.append(IoU_list)
                SE_list.append(se_list)
                SP_list.append(sp_list)
                AUC_list.append(auc_list)

                # test_loss_clus.append(loss_clu)
                # test_loss_nors.append(loss_nor)
                # test_loss_segs.append(loss_seg)

            # if phase == 'test' and epoch_loss_seg < best_loss:
            if phase == 'test' and epoch_loss_seg < best_loss:

                best_loss = epoch_loss_seg
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info("Best val loss {} save at epoch {}".format(best_loss, epoch + 1))
        logging.info(
            'Test: epoch {}:acc: {:.5f}, IoU: {:.5f}, F1 score: {:.5f}, SE:{:.5f}, SP:{:.5f}, AUC:{:.5f}'
            .format(epoch + 1, acc_list, IoU_list, F1_score_list, se_list, sp_list, auc_list))
    draw_loss(train_loss, test_loss, str(now))
    draw_f1(F1_list, str(now))
    draw_acc(Acc_list, str(now))
    draw_iou(IOU_list, str(now))
    create_dir('./saved')
    torch.save(best_model_wts, './saved/model_epoch:{}.pt'.format(best_epoch))
    logging.info(
        'Model saved. at {}'.format('./saved/model_spoch:{}_testloss:{}_{}.pt'.format(best_epoch, best_loss, str(now))))

    model.load_state_dict(best_model_wts)
    model.eval()

    dice_acc_test = 0
    dice_loss_test = DiceLoss(num_classes)

    with torch.no_grad():
        for i, d in enumerate(testloader, 0):
            img, instance_seg_mask, semantic_seg_mask, normal_edge_mask, cluster_edge_mask = d
            semantic_seg_mask2 = semantic_seg_mask.cpu().detach().numpy()
            normal_edge_mask2 = normal_edge_mask.cpu().detach().numpy()
            cluster_edge_mask2 = cluster_edge_mask.cpu().detach().numpy()
            # img = img.unsqueeze(0)
            img = img.float()
            img = img.to(device)

            output1, output2, output3 = model(img)

            d_l = dice_loss_test(output1, semantic_seg_mask.float(), softmax=True)
            dice_acc_test += 1 - d_l.item()

    # logging.info("dice_acc {}".format(dice_acc_test/dataset_sizes['test']))
    # draw_acc(dice_acc_test/dataset_sizes['test'],str('now'))


if __name__ == '__main__':
    main()
