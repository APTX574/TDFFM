import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import interp
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


num_class = 2


def print_plt(_score_list, _label_list, net_name, bz, is_p=True):
    score_array = np.array(_score_list)
    label_tensor = torch.tensor(_label_list, dtype=torch.int64)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    pre = np.where(score_array < 0.5, 0, 1)
    ax = pre - label_onehot
    ax1 = pre + label_onehot
    FN = float(np.sum(ax[:, 0] == 1))
    FP = float(np.sum(ax[:, 1] == 1))
    TN = float(np.sum(ax1[:, 0] == 2))
    TP = float(np.sum(ax1[:, 1] == 2))
    print("FN={}".format(FN))
    print("FP={}".format(FP))
    print("TN={}".format(TN))
    print("TN={}".format(TP))
    acc = (TP + TN) / (FN + FP + TP + TN)
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    print("precision" + str(precision))
    print("sensitivity" + str(sensitivity))
    F1 = 2 * precision * sensitivity / (precision + sensitivity)
    print("acc={}".format(acc))
    print("sensitivity={}".format(sensitivity))

    fontdict = {'family': 'Times New Roman', 'size': 16}
    if is_p is False:
        return

    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    label_onehot = np.array(label_onehot, dtype=int)


    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro

    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])


    plt.rc('font', family='Arial')
    plt.figure(dpi=350, figsize=(12, 12))

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    lw = 3
    print(acc)
    bwith = 4
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='ROC',
             color='deeppink', linestyle='-', linewidth=3, )

    precision1, recall1, _ = metrics.precision_recall_curve(label_onehot[:, 1], score_array[:, 1])
    aupr1 = metrics.auc(recall1, precision1)  # the value of roc_auc1
    # print(aupr1)
    plt.plot(recall1, precision1, 'b', label='PRC', linewidth=3,
             color='cornflowerblue')

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 01.01])
    plt.xlabel('FPR(or Recall)', fontsize=25, weight='bold')

    plt.ylabel('Sensitivity(or Precision)', fontsize=25,weight='bold' )
    plt.title('ROC Curve(or PR Curve) of Deep Forest', fontsize=25,weight='bold')
    plt.legend(loc="lower right", fontsize=25, )
    plt.text(0.70, 0.304, "AUC:{0:.3f}\n"
                          "AUPRC:{1:0.3f}\n"
                          "Accuracy:{2:0.3f}\n"
                          "Recall:{3:0.3f}"
                          # "F1:{4:0.2f}"
             .format(roc_auc_dict["micro"], aupr1, acc, sensitivity),
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec='grey', lw=1, alpha=0.5,
                       ), fontsize=25,fontproperties="Arial", )
    path = "image/{}/".format(net_name)
    if not os.path.exists(path):
        os.makedirs(path)
    print(roc_auc_dict["micro"])
    print(roc_auc_dict["macro"])
    plt.savefig('image/{}/rocx-{}-{}.jpg'.format(net_name, net_name, bz), dpi=350)
    plt.show()
