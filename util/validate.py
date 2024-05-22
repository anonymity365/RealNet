from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score,auc,average_precision_score
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_curve
from copy import deepcopy
from network.feature_transformer import get_TRR
import torch
import numpy as np

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres

def eval(testing_dataset_loader, model, device, v_id, uncalibrated_threshold):
    model.eval()
    y_true, y_pred = [], []
    tbar = tqdm(testing_dataset_loader)
    with torch.no_grad():
        for images, labels in tbar:
            images = images.to(device)
            labels = labels.to(device)

            y_pred.extend(model(images, device).flatten().detach().cpu().tolist())
            y_true.extend(labels.flatten().detach().cpu().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    ap = average_precision_score(y_true, y_pred)

    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])

    auc = roc_auc_score(y_true, y_pred)

    desc_score_indices = np.argsort(y_true, kind="mergesort")
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]
    best_threshold = find_best_threshold(y_true, y_pred)
    if v_id == 0:
        uncalibrated_threshold = best_threshold
    oracle_acc = accuracy_score(y_true, y_pred >= best_threshold)
    uncalibrated_acc = accuracy_score(y_true, y_pred >= uncalibrated_threshold)
    oracle_r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] >= best_threshold)
    oracle_f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] >= best_threshold)
    uncalibrated_r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] >= uncalibrated_threshold)
    uncalibrated_f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] >= uncalibrated_threshold)
    return [ap, uncalibrated_acc, oracle_acc, uncalibrated_r_acc, uncalibrated_f_acc, oracle_r_acc, oracle_f_acc,
            best_f1_score, auc, best_threshold, uncalibrated_threshold]
