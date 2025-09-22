import argparse
import time
import warnings

import numpy as np
import torch
torch.set_num_threads(8)
import torch.nn as nn
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch_geometric import seed_everything
from main_utils import get_data_split, setup_logging
from main_utils import imputation_, random_sample
from tensorize_normalize import tensorize_normalize_and_remove_part
from make_model import make_model
from models.model_base import count_parameters


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='P12', choices=['P12', 'P19', 'eICU', 'PAM'])
parser.add_argument('--with_missing_ratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0', action='store_true')
parser.add_argument('--model_name', type=str, default='SVMPT')
# parser.add_argument('--model_name', type=str, default='TransformerModelV2')
parser.add_argument('--seed', type=int, default=0, choices=list(range(10)))
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
parser.add_argument('--reverse', default=False, help='if True, use female, older for tarining; if False, use female or younger for training')
parser.add_argument('--feature_removal_level', type=str, default='sample', choices=['no_removal', 'set', 'sample'],
                    help='use this only when splittype==random; otherwise, set as no_removal')
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                    help='use this only with P12 dataset (mortality or length of stay)')
parser.add_argument('--imputation', type=str, default='no_imputation', choices=['no_imputation', 'mean', 'forward', 'cubic_spline'],
                    help='use this if you want to impute missing values')

args, unknown = parser.parse_known_args()


seed_everything(args.seed)


dataset = args.dataset
print('Dataset used: ', dataset)


if dataset in {'P12', 'P19', 'eICU'}:
    strategy = 2
elif dataset == 'PAM':
    strategy = 3


def main_loop(
    device, model, n_classes, criterion, optimizer, scheduler, batch_size, logger,
    TrainF, TrainS, TrainT, TrainY,
    ValF, ValS, ValT, ValY,
    TestF, TestS, TestT, TestY,
):
    idx_0 = np.where(TrainY == 0)[0]
    idx_1 = np.where(TrainY == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
    expanded_n1 = len(expanded_idx_1)

    if strategy == 1:
        n_batches = 10
    elif strategy == 2:
        K0 = n0 // int(batch_size / 2)
        K1 = expanded_n1 // int(batch_size / 2)
        n_batches = np.min([K0, K1])
    elif strategy == 3:
        n_batches = 30

    best_val_auroc = best_val_auprc = 0.0
    best_epoch = 0
    time_start = time.perf_counter()
    for epoch in range(num_epochs):
        elapsed_time = time.perf_counter() - time_start
        logger.info(f"Epoch idx: {epoch}")
        logger.info(f"Elapsed time: {elapsed_time}")
        training_loop(device, model, criterion, optimizer, batch_size, logger, TrainF, TrainS, TrainT, TrainY, idx_0, idx_1, expanded_idx_1, n_batches)
        improved, val_auroc, val_auprc = \
            validating_loop(model, device, ValF, ValT, ValS, ValY, best_val_auroc, n_classes)
        scheduler.step(val_auprc)

        if improved:
            best_epoch = epoch
            best_val_auroc = val_auroc
            best_val_auprc = val_auprc
            loss, accuracy, auroc, auprc, precision, recall, f1, cm, cls_report = \
                testing_loop(model, device, TestF, TestT, TestS, TestY, n_classes)

    logger.info("=" * 10 + "Final" + "=" * 10)
    logger.info(f"Best epoch:       {best_epoch}")
    logger.info(f"Test loss:        {loss}")
    logger.info(f"Test accuracy:    {accuracy * 100}")
    logger.info(f"Test auroc:       {auroc * 100}")
    logger.info(f"Test auprc:       {auprc * 100}")
    if dataset == 'PAM':
        logger.info(f"Test precision:   {precision * 100}")
        logger.info(f"Test recall:      {recall * 100}")
        logger.info(f"Test f1:          {f1 * 100}")

    logger.info(f"Test confusion matrix:\n{cm}")
    logger.info(f"Test classification report:\n{cls_report}\n")
    return best_epoch, best_val_auroc, best_val_auprc, \
        loss, accuracy, auroc, auprc, precision, recall, f1


def training_loop(device, model, criterion, optimizer, batch_size, logger, TrainF, TrainS, TrainT, TrainY, idx_0, idx_1, expanded_idx_1, n_batches):
    model.train()
    if strategy == 2:
        np.random.shuffle(expanded_idx_1)
        I1 = expanded_idx_1
        np.random.shuffle(idx_0)
        I0 = idx_0

    trainY_list = []
    output_list = []
    for n in range(n_batches):
        if strategy == 1:
            idx = random_sample(idx_0, idx_1, batch_size)
        elif strategy == 2:
            idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
            idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
            idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
        elif strategy == 3:
            idx = np.random.choice(list(range(TrainF.shape[1])), size=int(batch_size), replace=False)
            # idx = random_sample_8(ytrain, batch_size)   # to balance dataset

        if dataset in {'P12', 'P19', 'eICU'}:
            trainF, trainT, trainS, trainY = TrainF[:, idx, :], TrainT[:, idx, :], TrainS[idx], TrainY[idx]
            trainF, trainT, trainS, trainY = trainF.to(device), trainT.to(device), trainS.to(device), trainY.to(device)
        elif dataset == 'PAM':
            trainF, trainT, trainS, trainY = TrainF[:, idx, :], TrainT[:, idx, :], None, TrainY[idx]
            trainF, trainT, trainS, trainY = trainF.to(device), trainT.to(device), None, trainY.to(device)

        lengths = torch.sum(trainT > 0, dim=0)
        outputs = model.forward(trainF, trainT, trainS, lengths)
        optimizer.zero_grad()
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()

        trainY_list.append(trainY.detach().cpu())
        output_list.append(outputs.detach().cpu())

    trainY_all = torch.cat(trainY_list)
    output_all = torch.cat(output_list)

    precision, recall, f1 = None, None, None
    if dataset in {'P12', 'P19', 'eICU'}:
        trainP_all = torch.squeeze(torch.sigmoid(output_all))
        auroc = roc_auc_score(trainY_all, trainP_all[:, 1])
        auprc = average_precision_score(trainY_all, trainP_all[:, 1])
    elif dataset == 'PAM':
        trainP_all = torch.squeeze(nn.functional.softmax(output_all, dim=1))
        trainY_all = one_hot(trainY_all)
        auroc = roc_auc_score(trainY_all, trainP_all)
        auprc = average_precision_score(trainY_all, trainP_all)
        precision = precision_score(trainY_all, trainP_all, average='macro', )
        recall = recall_score(trainY_all, trainP_all, average='macro', )
        f1 = f1_score(trainY_all, trainP_all, average='macro', )

    loss = criterion(output_all, trainY_all).item()
    trainC_all = torch.argmax(trainP_all, 1)
    accuracy = torch.sum(trainY_all.ravel() == trainC_all.ravel()).item() / trainY_all.shape[0]
    cls_report = classification_report(trainY_all, trainC_all)
    cm = confusion_matrix(trainY_all, trainC_all, labels=[0, 1])

    logger.info(f"Train loss:       {loss}")
    logger.info(f"Train accuracy:   {accuracy * 100}")
    logger.info(f"Train auroc:      {auroc * 100}")
    logger.info(f"Train auprc:      {auprc * 100}")
    if dataset == 'PAM':
        logger.info(f"Train precision:  {precision * 100}")
        logger.info(f"Train recall:     {recall * 100}")
        logger.info(f"Train f1:         {f1 * 100}")

    logger.info(f"Train confusion matrix:\n{cm}")
    logger.info(f"Train classification report:\n{cls_report}\n")


@torch.no_grad()
def validating_loop(model, device, valF, valT, valS, valY, best_auroc, n_classes):
    output = forward(model, device, valF, valT, valS, n_classes)
    output = torch.squeeze(torch.sigmoid(output))
    output = output.detach().cpu()
    if dataset in {'P12', 'P19', 'eICU'}:
        valP = torch.squeeze(torch.sigmoid(output))
        auroc = roc_auc_score(valY, valP[:, 1])
        auprc = average_precision_score(valY, valP[:, 1])
    elif dataset == 'PAM':
        valP = torch.squeeze(nn.functional.softmax(output, dim=1))
        valY = one_hot(valY)
        auroc = roc_auc_score(valY, valP)
        auprc = average_precision_score(valY, valP)
        precision = precision_score(valY, valP, average='macro', )
        recall = recall_score(valY, valP, average='macro', )
        f1 = f1_score(valY, valP, average='macro', )

    loss = criterion(output, valY).item()
    valC = torch.argmax(valP, 1)
    accuracy = torch.sum(valY.ravel() == valC.ravel()).item() / valY.shape[0]
    cls_report = classification_report(valY, valC)
    cm = confusion_matrix(valY, valC, labels=[0, 1])

    improved = False
    if auroc > best_auroc:
        improved = True
        logger.info(f"Val loss:         {loss}")
        logger.info(f"Val accuracy:     {accuracy * 100}")
        logger.info(f"Val auroc:        {auroc * 100}")
        logger.info(f"Val auprc:        {auprc * 100}")
        if dataset == 'PAM':
            logger.info(f"Val precision:    {precision * 100}")
            logger.info(f"Val recall:       {recall * 100}")
            logger.info(f"Val f1:           {f1 * 100}")

        logger.info(f"Val confusion matrix:\n{cm}")
        logger.info(f"Val classification report:\n{cls_report}\n")

    return improved, auroc, auprc


@torch.no_grad()
def testing_loop(model, device, testF, testT, testS, testY, n_classes):
    model.eval()
    output = forward(model, device, testF, testT, testS, n_classes)
    output = torch.squeeze(torch.sigmoid(output))
    if dataset in {'P12', 'P19', 'eICU'}:
        testP = torch.squeeze(torch.sigmoid(output))
        auroc = roc_auc_score(testY, testP[:, 1])
        auprc = average_precision_score(testY, testP[:, 1])
        precision = recall = f1 = None
    elif dataset == 'PAM':
        testP = torch.squeeze(nn.functional.softmax(output, dim=1))
        testY = one_hot(testY)
        auroc = roc_auc_score(testY, testP)
        auprc = average_precision_score(testY, testP)
        precision = precision_score(testY, testP, average='macro', )
        recall = recall_score(testY, testP, average='macro', )
        f1 = f1_score(testY, testP, average='macro', )

    loss = criterion(output, testY).item()
    testC = torch.argmax(testP, 1)
    accuracy = torch.sum(testY.ravel() == testC.ravel()).item() / testY.shape[0]
    cls_report = classification_report(testY, testC)
    cm = confusion_matrix(testY, testC, labels=[0, 1])

    logger.info(f"Test loss:        {loss}")
    logger.info(f"Test accuracy:    {accuracy * 100}")
    logger.info(f"Test auroc:       {auroc * 100}")
    logger.info(f"Test auprc:       {auprc * 100}")
    if dataset == 'PAM':
        logger.info(f"Test precision:   {precision * 100}")
        logger.info(f"Test recall:      {recall * 100}")
        logger.info(f"Test f1:          {f1 * 100}")

    logger.info(f"Test confusion matrix:\n{cm}")
    logger.info(f"Test classification report:\n{cls_report}")

    return loss, accuracy, auroc, auprc, precision, recall, f1, cm, cls_report


def forward(model, device, xF, xT, xS, n_classes):
    model.eval()
    T, N, Ff = xF.shape
    n_batches, rem = N // batch_size, N % batch_size
    outs = torch.zeros(N, n_classes)

    start = 0
    xs = None
    for i in range(n_batches):
        xf = xF[:, start:start + batch_size, :].to(device)
        xt = xT[:, start:start + batch_size, :].to(device)
        if xS is not None:
            xs = xS[start:start + batch_size].to(device)
        lengths = torch.sum(xt > 0, dim=0)
        out = model.forward(xf, xt, xs, lengths)
        outs[start:start + batch_size] = out.detach().cpu()
        start += batch_size
    if rem > 0:
        xf = xF[:, start:, :].to(device)
        xt = xT[:, start:, :].to(device)
        if xS is not None:
            xs = xS[start:].to(device)
        lengths = torch.sum(xt > 0, dim=0)
        out = model.forward(xf, xt, xs, lengths)
        outs[start:] = out.detach().cpu()
    return outs


def one_hot(y_: np.ndarray):
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


# ############# main loop ####################
n_splits = 5
num_epochs = 30
batch_size = 128
feature_removal_level = args.feature_removal_level   # possible values: 'sample', 'set'

if args.with_missing_ratio == True:
    missing_ratios = [0.1, 0.3, 0.5]
else:
    missing_ratios = [0]

n_runs = len(missing_ratios)


if args.dataset in {'P12', 'P19'}:
    n_classes = 2

for run_idx in range(n_runs):
    best_epoch_list, best_val_auroc_list, best_val_auprc_list, \
        loss_list, accuracy_list, auroc_list, auprc_list, \
            precision_list, recall_list, f1_list = [[] for _ in range(10)]

    missing_ratio = missing_ratios[run_idx]
    logger = setup_logging(dataset, args.model_name, args.seed, missing_ratio)
    logger.info(f'===================== Begin (Run Id: {run_idx}) =====================')

    for split_idx in range(n_splits):
        logger.info(f"===== Split Id: {split_idx} ======")
        logger.info(f"Missing Ratio: {missing_ratio}")

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cpu()
        model = make_model(dataset, args.model_name).to(device)
        count_parameters(logger, model)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1,
            patience=1, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True
        )
        Ptrain, Pval, Ptest, TrainY, Yval, Ytest = get_data_split(dataset, logger)
        imputation_(args.imputation, dataset, Ptrain, Pval, Ptest)

        TrainF, TrainS, TrainT, TrainY, \
            ValF, ValS, ValT, ValY, \
            TestF, TestS, TestT, TestY = \
            tensorize_normalize_and_remove_part(
                dataset, Ptrain, Pval, Ptest, TrainY, Yval, Ytest,
                feature_removal_level, missing_ratio
            )
        TrainF = TrainF.permute((1, 0, 2))
        TrainT = TrainT.permute((1, 0, 2))
        ValF = ValF.permute((1, 0, 2))
        ValT = ValT.permute((1, 0, 2))
        TestF = TestF.permute((1, 0, 2))
        TestT = TestT.permute((1, 0, 2))

        best_epoch, best_val_auroc, best_val_auprc, \
            loss, accuracy, auroc, auprc, precision, recall, f1 = main_loop(
                device, model, n_classes, criterion, optimizer, scheduler, batch_size, logger,
                TrainF, TrainS, TrainT, TrainY,
                ValF, ValS, ValT, ValY,
                TestF, TestS, TestT, TestY
            )

        best_epoch_list.append(best_epoch)
        best_val_auroc_list.append(best_val_auroc)
        best_val_auprc_list.append(best_val_auprc)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    min_epoch, max_epoch = min(best_epoch_list), max(best_epoch_list)
    mean_epoch, std_epoch = np.mean(best_epoch_list) * 100, np.std(best_epoch_list) * 100
    mean_val_auroc, std_val_auroc = np.mean(best_val_auroc_list) * 100, np.std(best_val_auroc_list) * 100
    mean_val_auprc, std_val_auprc = np.mean(best_val_auprc_list) * 100, np.std(best_val_auprc_list) * 100
    mean_loss, std_loss = np.mean(loss_list) * 100, np.std(loss_list) * 100
    mean_accuracy, std_accuracy = np.mean(accuracy_list) * 100, np.std(accuracy_list) * 100
    mean_auroc, std_auroc = np.mean(auroc_list) * 100, np.std(auroc_list) * 100
    mean_auprc, std_auprc = np.mean(auprc_list) * 100, np.std(auprc_list) * 100

    logger.info(f'===================== Summary (Run Id: {run_idx}) =====================')
    logger.info(f'Epoch           = {min_epoch}, {max_epoch}, {mean_epoch:.2f}  +/- {std_epoch:.2f}')
    logger.info(f'Val  AUROC      = {mean_val_auroc:.2f}  +/- {std_val_auroc:.2f}')
    logger.info(f'Val  AUPRC      = {mean_val_auprc:.2f}  +/- {std_val_auprc:.2f}')
    logger.info(f'Test loss       = {mean_loss:.2f}  +/- {std_loss:.2f}')
    logger.info(f'Test Accuracy   = {mean_accuracy:.2f}  +/- {std_accuracy:.2f}')
    logger.info(f'Test AUROC      = {mean_auroc:.2f}  +/- {std_auroc:.2f}')
    logger.info(f'Test AUPRC      = {mean_auprc:.2f}  +/- {std_auprc:.2f}')

    if dataset == 'PAM':
        mean_precision, std_precision = np.mean(precision_list) * 100, np.std(precision_list) * 100
        mean_recall, std_recall = np.mean(recall_list) * 100, np.std(recall_list) * 100
        mean_f1, std_f1 = np.mean(f1_list) * 100, np.std(f1_list) * 100
        logger.info(f'Test Precision  = {mean_precision:.2f}  +/- {std_precision:.2f}')
        logger.info(f'Test Recall     = {mean_recall:.2f}  +/- {std_recall:.2f}')
        logger.info(f'Test F1         = {mean_f1:.2f}  +/- {std_f1:.2f}')
    
    logger.info("\n")
    logger.info("=" * 8 + "done" + "=" * 8)
    logger.info("\n" * 3)
