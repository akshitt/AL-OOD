import h5py
import time
import random
import datetime
import copy
import numpy as np
import os
import csv
import json
import subprocess
import sys
import PIL.Image as Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision.datasets import cifar
from matplotlib import pyplot as plt
sys.path.append('/mnt/data2/akshit/distil/')
sys.path.append('/mnt/data2/akshit/trust/')
from distil.utils.models.resnet import ResNet18
from trust.utils.pneumoniamnist import load_dataset_custom_1 as load_dataset_custom
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torch.utils.data import Subset
from torch.autograd import Variable
import tqdm
from math import floor
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from distil.active_learning_strategies.scmi import SCMI
from distil.active_learning_strategies.smi import SMI
from distil.active_learning_strategies.badge import BADGE
from distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.active_learning_strategies.gradmatch_active import GradMatchActive
from distil.active_learning_strategies.glister import GLISTER
from trust.strategies.random_sampling import RandomSampling

seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
from distil.utils.utils import *

feature = "ood"
device_id = 0
run="fkna_3"
datadir = 'data/'
data_name = 'cifar10'
model_name = 'ResNet18'
num_rep = 10
learning_rate = 0.01
num_runs = 1  # number of random runs
computeClassErrorLog = True

magnification = 1
device = "cuda:"+str(device_id) if torch.cuda.is_available() else "cpu"
datkbuildPath = "./datk/build"
exePath = "cifarSubsetSelector"
print("Using Device:", device)
doublePrecision = True
linearLayer = True
miscls = True
# handler = DataHandler_CIFAR10
augTarget = True
embedding_type = "gradients"

num_cls=2
budget=5
num_epochs = int(10)
split_cfg = {'num_cls_idc':2, 'per_idc_train':5, 'per_idc_val':15, 'per_idc_lake':2500, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':5000}

initModelPath = "/mnt/data2/akshit/Pneumonia/weights/" + data_name + "_" + feature + "_" + model_name + "_" + str(learning_rate) + "_" + str(split_cfg["per_idc_train"]) + "_" + str(split_cfg["per_idc_val"]) + "_" + str(split_cfg["num_cls_idc"])


#Functions
def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss

def init_weights(m):
#     torch.manual_seed(35)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
                
def create_model(name, num_cls, device, embedding_type):
    if name == 'ResNet18':
        if embedding_type == "gradients":
            model = ResNet18(num_cls)
        else:
            model = models.resnet18()
    elif name == 'MnistNet':
        model = MnistNet()
    elif name == 'ResNet164':
        model = ResNet164(num_cls)
    model.apply(init_weights)
    model = model.to(device)
    return model

def loss_function():
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    return criterion, criterion_nored

def optimizer_with_scheduler(model, num_epochs, learning_rate, m=0.9, wd=5e-4):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=m, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler

def optimizer_without_scheduler(model, learning_rate, m=0.9, wd=5e-4):
#     optimizer = optim.Adam(model.parameters(),weight_decay=wd)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=m, weight_decay=wd)
    return optimizer

def generate_cumulative_timing(mod_timing):
    tmp = 0
    mod_cum_timing = np.zeros(len(mod_timing))
    for i in range(len(mod_timing)):
        tmp += mod_timing[i]
        mod_cum_timing[i] = tmp
    return mod_cum_timing/3600

def find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, 
                       final_tst_predictions, saveDir, prefix):
    #find queries from the validation set that are erroneous
#     saveDir = os.path.join(saveDir, prefix)
#     if(not(os.path.exists(saveDir))):
#         os.mkdir(saveDir)
    val_err_idx = list(np.where(np.array(final_val_classifications) == False)[0])
    tst_err_idx = list(np.where(np.array(final_tst_classifications) == False)[0])
    val_class_err_idxs = []
    tst_err_log = []
    val_err_log = []
    for i in range(num_cls):
        if(feature=="ood"): tst_class_idxs = list(torch.where(torch.Tensor(test_set.targets.float()) == i)[0].cpu().numpy())
        if(feature=="classimb"): tst_class_idxs = list(torch.where(torch.Tensor(test_set.targets) == i)[0].cpu().numpy())
        val_class_idxs = list(torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy())
        #err classifications per class
        val_err_class_idx = set(val_err_idx).intersection(set(val_class_idxs))
        tst_err_class_idx = set(tst_err_idx).intersection(set(tst_class_idxs))
        if(len(val_class_idxs)>0):
            val_error_perc = round((len(val_err_class_idx)/len(val_class_idxs))*100,2)
        else:
            val_error_perc = 0
            
        tst_error_perc = round((len(tst_err_class_idx)/len(tst_class_idxs))*100,2)
        print("val, test error% for class ", i, " : ", val_error_perc, tst_error_perc)
        val_class_err_idxs.append(val_err_class_idx)
        tst_err_log.append(tst_error_perc)
        val_err_log.append(val_error_perc)
    tst_err_log.append(sum(tst_err_log)/len(tst_err_log))
    val_err_log.append(sum(val_err_log)/len(val_err_log))
    return tst_err_log, val_err_log, val_class_err_idxs

def aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget, augrandom=False):
    all_lake_idx = list(range(len(lake_set)))
    if(not(len(subset)==budget) and augrandom):
        print("Budget not filled, adding ", str(int(budget) - len(subset)), " randomly.")
        remain_budget = int(budget) - len(subset)
        remain_lake_idx = list(set(all_lake_idx) - set(subset))
        random_subset_idx = list(np.random.choice(np.array(remain_lake_idx), size=int(remain_budget), replace=False))
        subset += random_subset_idx
    lake_ss = SubsetWithTargets(true_lake_set, subset, torch.Tensor(true_lake_set.targets.float())[subset])
    if(feature=="ood"): 
        ood_lake_idx = list(set(lake_subset_idxs)-set(subset))
        private_set =  SubsetWithTargets(true_lake_set, ood_lake_idx, torch.Tensor(np.array([split_cfg['num_cls_idc']]*len(ood_lake_idx))).float())
    remain_lake_idx = list(set(all_lake_idx) - set(lake_subset_idxs))
    remain_lake_set = SubsetWithTargets(lake_set, remain_lake_idx, torch.Tensor(lake_set.targets.float())[remain_lake_idx])
    remain_true_lake_set = SubsetWithTargets(true_lake_set, remain_lake_idx, torch.Tensor(true_lake_set.targets.float())[remain_lake_idx])
    print(len(lake_ss),len(remain_lake_set),len(lake_set))
    if(feature!="ood"): assert((len(lake_ss)+len(remain_lake_set))==len(lake_set))
    aug_train_set = torch.utils.data.ConcatDataset([train_set, lake_ss])
    if(feature=="ood"): 
        return aug_train_set, remain_lake_set, remain_true_lake_set, private_set, lake_ss
    else:
        return aug_train_set, remain_lake_set, remain_true_lake_set, lake_ss
                        
def getQuerySet(val_set, val_class_err_idxs, imb_cls_idx, miscls):
    miscls_idx = []
    if(miscls):
        for i in range(len(val_class_err_idxs)):
            if i in imb_cls_idx:
                miscls_idx += val_class_err_idxs[i]
        print("total misclassified ex from imb classes: ", len(miscls_idx))
    else:
        for i in imb_cls_idx:
            imb_cls_samples = list(torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy())
            miscls_idx += imb_cls_samples
        print("total samples from imb classes as targets: ", len(miscls_idx))
    return Subset(val_set, miscls_idx)

def getPrivateSet(lake_set, subset, private_set):
    #augment prev private set and current subset
    new_private_set = SubsetWithTargets(lake_set, subset, torch.Tensor(lake_set.targets.float())[subset])
#     new_private_set =  Subset(lake_set, subset)
    total_private_set = torch.utils.data.ConcatDataset([private_set, new_private_set])
    return total_private_set

def remove_ood_points(lake_set, subset, idc_idx):
    idx_subset = []
    subset_cls = torch.Tensor(lake_set.targets.float())[subset]
    for i in idc_idx:
        idc_subset_idx = list(torch.where(subset_cls == i)[0].cpu().numpy())
        idx_subset += list(np.array(subset)[idc_subset_idx])
    print(len(idx_subset),"/",len(subset), " idc points.")
    return idx_subset

def getPerClassSel(lake_set, subset, num_cls):
    perClsSel = []
    subset_cls = torch.Tensor(lake_set.targets.float())[subset]
    for i in range(num_cls):
        cls_subset_idx = list(torch.where(subset_cls == i)[0].cpu().numpy())
        perClsSel.append(len(cls_subset_idx))
    return perClsSel


def train_model_al(datkbuildPath, exePath, num_epochs, dataset_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run,
                device, computeErrorLog, strategy="SIM", sf=""):
#     torch.manual_seed(42)
#     np.random.seed(42)
    print(strategy, sf)
    #load the dataset based on type of feature
    train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls = load_dataset_custom(datadir, feature, split_cfg, False, True)
    print("selected classes are: ", sel_cls_idx)

    if(feature=="ood"): num_cls+=1 #Add one class for OOD class
    N = len(train_set)
    trn_batch_size = 20
    val_batch_size = 10
    tst_batch_size = 100

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=trn_batch_size,
                                              shuffle=True, pin_memory=True)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, 
                                            shuffle=False, pin_memory=True)

    tstloader = torch.utils.data.DataLoader(test_set, batch_size=tst_batch_size,
                                             shuffle=False, pin_memory=True)
    
    lakeloader = torch.utils.data.DataLoader(lake_set, batch_size=tst_batch_size,
                                         shuffle=False, pin_memory=True)
    true_lake_set = copy.deepcopy(lake_set)
    # Budget for subset selection
    bud = budget
   
    # Variables to store accuracies
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    tst_losses = np.zeros(num_epochs)
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    full_trn_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    final_tst_predictions = []
    final_tst_classifications = []
    best_val_acc = -1
    csvlog = []
    val_csvlog = []
    # Results logging file
    print_every = 3
#     all_logs_dir = '/content/drive/MyDrive/research/tdss/SMI_active_learning_results_woVal/' + dataset_name  + '/' + feature + '/'+  sf + '/' + str(bud) + '/' + str(run)
    all_logs_dir = './SMI_active_learning_results/' + dataset_name  + '/' + feature + '/'+  sf + '/' + str(bud) + '/' + str(run)
    print("Saving results to: ", all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir])
    exp_name = dataset_name + "_" + feature +  "_" + strategy + "_" + str(len(sel_cls_idx))  +"_" + sf +  '_budget:' + str(bud) + '_epochs:' + str(num_epochs) + '_linear:'  + str(linearLayer) + '_runs' + str(run)
    print(exp_name)
    res_dict = {"dataset":data_name, 
                "feature":feature, 
                "sel_func":sf,
                "sel_budget":budget, 
                "num_selections":num_epochs, 
                "model":model_name, 
                "learning_rate":learning_rate, 
                "setting":split_cfg, 
                "all_class_acc":None, 
                "test_acc":[],
                "sel_per_cls":[], 
                "sel_cls_idx":sel_cls_idx.tolist()}
    # Model Creation
    model = create_model(model_name, num_cls, device, embedding_type)
    model1 = create_model(model_name, num_cls, device, embedding_type)
    
    # Loss Functions
    criterion, criterion_nored = loss_function()
    
    strategy_args = {'batch_size': 20, 'device':'cuda', 'num_partitions':1, 'wrapped_strategy_class': None, 
         'embedding_type':'gradients', 'keep_embedding':False, 'budget':'budget'}
    unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
    if(strategy == "AL"):
        if(sf=="badge"):
            strategy_sel = BADGE(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf=="us"):
            strategy_sel = EntropySampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf=="glister" or sf=="glister-tss"):
            strategy_sel = GLISTER(train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set, typeOf='rand', lam=0.1)
        elif(sf=="gradmatch-tss"):
            strategy_sel = GradMatchActive(train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set)
        elif(sf=="coreset"):
            strategy_sel = CoreSet(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf=="leastconf"):
            strategy_sel = LeastConfidence(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf=="margin"):
            strategy_sel = MarginSampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
    if(strategy == "SIM"):
        if(sf.endswith("mic")):
            strategy_args['scmi_function'] = sf.split("mic")[0] + "cmi"
            strategy_sel = SCMI(train_set, unlabeled_lake_set, val_set, val_set, model, num_cls, strategy_args)
        elif(sf.endswith("mi")):
            strategy_args['smi_function'] = sf
            strategy_sel = SMI(train_set, unlabeled_lake_set, val_set, model, num_cls, strategy_args)
    if(strategy == "random"):
        strategy_sel = RandomSampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        
        strategy_args['verbose'] = False
        strategy_args['optimizer'] = "LazyGreedy"

    # Getting the optimizer and scheduler
#     optimizer, scheduler = optimizer_with_scheduler(model, num_epochs, learning_rate)
    optimizer = optimizer_without_scheduler(model, learning_rate)
    private_set = []

    for i in range(num_epochs):
        print("AL epoch: ", i)
        tst_loss = 0
        tst_correct = 0
        tst_total = 0
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        if(i==0):
            print("initial training epoch")
            if(os.path.exists(initModelPath)):
                model.load_state_dict(torch.load(initModelPath, map_location=device))
                print("Init model loaded from disk, skipping init training: ", initModelPath)
                model.eval()
                with torch.no_grad():
                    final_val_predictions = []
                    final_val_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(valloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        if(feature=="ood"): 
                            _, predicted = outputs[...,:-1].max(1)
                        else:
                            _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                        final_val_predictions += list(predicted.cpu().numpy())
                        final_val_classifications += list(predicted.eq(targets).cpu().numpy())
  
                    final_tst_predictions = []
                    final_tst_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(tstloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        tst_loss += loss.item()
                        if(feature=="ood"): 
                            _, predicted = outputs[...,:-1].max(1)
                        else:
                            _, predicted = outputs.max(1)
                        tst_total += targets.size(0)
                        tst_correct += predicted.eq(targets).sum().item()
                        final_tst_predictions += list(predicted.cpu().numpy())
                        final_tst_classifications += list(predicted.eq(targets).cpu().numpy())                
                    best_val_acc = (val_correct/val_total)
                    val_acc[i] = val_correct / val_total
                    tst_acc[i] = tst_correct / tst_total
                    val_losses[i] = val_loss
                    tst_losses[i] = tst_loss
                    res_dict["test_acc"].append(tst_acc[i])
                continue
        else:
            unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
            strategy_sel.update_data(train_set, unlabeled_lake_set)
            #compute the error log before every selection
            if(computeErrorLog):
                tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, final_tst_predictions, all_logs_dir, sf+"_"+str(bud))
                csvlog.append(tst_err_log)
                val_csvlog.append(val_err_log)
            ####SIM####
            if(strategy=="SIM" or strategy=="SF"):
                if(sf.endswith("mi")):
                    if(feature=="classimb"):
                        #make a dataloader for the misclassifications - only for experiments with targets
                        miscls_set = getQuerySet(val_set, val_class_err_idxs, sel_cls_idx, miscls)
                        strategy_sel.update_queries(miscls_set)
                elif(sf.endswith("mic")): #configured for the OOD setting
                    print("val set targets: ", val_set.targets)
                    strategy_sel.update_queries(val_set) #In-dist samples are in Val 
                    if(len(private_set)!=0):
                        print("private set targets: ", private_set.targets)
                        strategy_sel.update_privates(private_set)

            ###AL###
            elif(strategy=="AL"):
                if(sf=="glister-tss" or sf=="gradmatch-tss"):
                    miscls_set = getQuerySet(val_set, val_class_err_idxs, sel_cls_idx, miscls)
                    strategy_sel.update_queries(miscls_set)
                    print("reinit AL with targeted miscls samples")
                
            elif(strategy=="random"):
                subset = np.random.choice(np.array(list(range(len(lake_set)))), size=budget, replace=False)
            
            strategy_sel.update_model(model)
            subset = strategy_sel.select(budget)
#             print("True targets of subset: ", torch.Tensor(true_lake_set.targets.float())[subset])
#             hypothesized_targets = strategy_sel.predict(unlabeled_lake_set)
#             print("Hypothesized targets of subset: ", hypothesized_targets)
            lake_subset_idxs = subset #indices wrt to lake that need to be removed from the lake
            if(feature=="ood"): #remove ood points from the subset
                subset = remove_ood_points(true_lake_set, subset, sel_cls_idx)
            
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            perClsSel = getPerClassSel(true_lake_set, lake_subset_idxs, num_cls)
            res_dict['sel_per_cls'].append(perClsSel)
            
            #augment the train_set with selected indices from the lake
            if(feature=="classimb"):
                train_set, lake_set, true_lake_set, add_val_set = aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget, True) #aug train with random if budget is not filled
                if(augTarget): val_set = ConcatWithTargets(val_set, add_val_set)
            elif(feature=="ood"):
                train_set, lake_set, true_lake_set, new_private_set, add_val_set = aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget)
                train_set = torch.utils.data.ConcatDataset([train_set, new_private_set]) #Add the OOD samples with a common OOD class
                val_set = ConcatWithTargets(val_set, add_val_set)
                if(len(private_set)!=0):
                    private_set = ConcatWithTargets(private_set, new_private_set)
                else:
                    private_set = new_private_set
            else:
                train_set, lake_set, true_lake_set = aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget)
            print("After augmentation, size of train_set: ", len(train_set), " lake set: ", len(lake_set), " val set: ", len(val_set))
    
#           Reinit train and lake loaders with new splits and reinit the model
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=trn_batch_size, shuffle=True, pin_memory=True)
            lakeloader = torch.utils.data.DataLoader(lake_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True)

            if(augTarget):
              valloader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False, pin_memory=True)
            model = create_model(model_name, num_cls, device, strategy_args['embedding_type'])
            optimizer = optimizer_without_scheduler(model, learning_rate)
                
        #Start training
        start_time = time.time()
        num_ep=1
        while(full_trn_acc[i]<0.99 and num_ep<300):
            model.train()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                # Variables in Pytorch are differentiable.
                inputs, target = Variable(inputs), Variable(inputs)
                # This will zero out the gradients for this batch.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
#             scheduler.step()
          
            full_trn_loss = 0
            full_trn_correct = 0
            full_trn_total = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    full_trn_loss += loss.item()
                    _, predicted = outputs.max(1)
                    full_trn_total += targets.size(0)
                    full_trn_correct += predicted.eq(targets).sum().item()
                full_trn_acc[i] = full_trn_correct / full_trn_total
                print("Selection Epoch ", i, " Training epoch [" , num_ep, "]" , " Training Acc: ", full_trn_acc[i], end="\r")
                num_ep+=1
            timing[i] = time.time() - start_time
        with torch.no_grad():
            final_val_predictions = []
            final_val_classifications = []
            for batch_idx, (inputs, targets) in enumerate(valloader): #Compute Val accuracy
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                if(feature=="ood"): 
                    _, predicted = outputs[...,:-1].max(1)
                else:
                    _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                final_val_predictions += list(predicted.cpu().numpy())
                final_val_classifications += list(predicted.eq(targets).cpu().numpy())

            final_tst_predictions = []
            final_tst_classifications = []
            for batch_idx, (inputs, targets) in enumerate(tstloader): #Compute test accuracy
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                if(feature=="ood"): 
                    _, predicted = outputs[...,:-1].max(1)
                else:
                    _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()
                final_tst_predictions += list(predicted.cpu().numpy())
                final_tst_classifications += list(predicted.eq(targets).cpu().numpy())                
            val_acc[i] = val_correct / val_total
            tst_acc[i] = tst_correct / tst_total
            val_losses[i] = val_loss
            fulltrn_losses[i] = full_trn_loss
            tst_losses[i] = tst_loss
            full_val_acc = list(np.array(val_acc))
            full_timing = list(np.array(timing))
            res_dict["test_acc"].append(tst_acc[i])
            print('Epoch:', i + 1, 'FullTrn,TrainAcc,ValLoss,ValAcc,TstLoss,TstAcc,Time:', full_trn_loss, full_trn_acc[i], val_loss, val_acc[i], tst_loss, tst_acc[i], timing[i])
        if(i==0):
            torch.save(model.state_dict(), initModelPath) #save initial train model if not present
    if(computeErrorLog):
        tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, final_tst_predictions, all_logs_dir, sf+"_"+str(bud))
        csvlog.append(tst_err_log)
        val_csvlog.append(val_err_log)
        print(csvlog)
        res_dict["all_class_acc"] = csvlog
        res_dict["all_val_class_acc"] = val_csvlog
        with open(os.path.join(all_logs_dir, exp_name+".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(csvlog)
    #save results dir with test acc and per class selections
    with open(os.path.join(all_logs_dir, exp_name+".json"), 'w') as fp:
        json.dump(res_dict, fp)
    plt.xlabel('AL epochs')
    plt.ylabel('Validation Accuracy')
    plt.plot(val_acc, label=f'{strategy}-{sf}')
    plt.title('Budget:'+str(budget)+'  Trainset:'+ str(split_cfg['per_idc_train']))

        
# LOGDETCMI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'logdetmic')

## FLCMI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'flmic')

# FL2MI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'fl2mi')

# FL1MI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'fl1mi')

# BADGE
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","badge")

# US
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","us")

# # GLISTER
# train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","glister-tss")

# # GCMI+DIV
# train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'div-gcmi')

# GCMI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'gcmi')

# LOGDETMI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'logdetmi')

# Random
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "random",'random')

# # CORESET
# train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","coreset")
# # LEASTCONF
# train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","leastconf")

# # MARGIN SAMPLING
# train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","margin")


plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.savefig('Budget:'+str(budget)+'  Trainset:'+ str(split_cfg['per_idc_train']))
plt.clf()

# # FL
# train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SF",'fl')

# # LOGDET
# train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SF",'logdet')


