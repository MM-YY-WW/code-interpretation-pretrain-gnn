#是python的命令行解析的标准模块，内置于python，
#可以直接在命令行中就可以向程序中传入参数并让程序运行
#
import argparse


from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#用来显示进度条的而且基本不影响原程序效率
from tqdm import tqdm
import numpy as np


from model import GNN, GNN_graphpred

#用的sklearn里面的ROC AUC
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os

#shell utility的缩写，实现了在python中实现文件复制、移动、压缩、解压等高级功能，是python的系统模块
import shutil

#虽然在pytorch下面但是内核是tensorflow里面的board，所以安装之前得先安装tensorflow
#summarywriter就相当于一个日志，保存了要做图的所有信息
from tensorboardX import SummaryWriter

#与BCELoss()不同的是网络的输出无需用sigmoid压缩，函数内部整合了nn.sigmoid()和
#nn.BCELoss()并且使用log-sum-exp trick提高了函数的稳定性。同样可用于二分类及多分类标签
criterion = nn.BCEWithLogitsLoss(reduction = "none")

#train函数，传入argparse里面写的args，选好的model，device，数据的loader和optimizer
def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



def main():
    # Training settings
    #设置解析器 ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    #为解析器设置参数信息，以告诉解析器命令行字符串中的哪些内容应解析为那些类型的对象
    
    #如果有多个gpu选哪个，默认=0
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    
    #batch size设定为多少，默认=32
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    
    #跑多少个epoch，默认=100
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    
    #learning rate多少，默认=0.001
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    
    #特征提取层的相对学习率？
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    
    #权值衰减，为了防止过拟合的，这个事放在正则前面的一个系数，正则项一般指示模型的复杂度
    #权值衰减的作用是调节模型复杂度对于损失函数的影响，如果weight decay很大，则复杂的模型的损失函数的值也就越大
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    
    #GNN的消息传递层有多少层，默认=5
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    
    #默认embedding 维度是300维
    parser.add_argument('--emb_dim', type=int, default=300,           
                        help='embedding dimensions (default: 300)')
    
    #默认drop out=0.5
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    
    #图池化的方法，默认=mean
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    
    #不同层的节点特征如何组合在一起，last是啥组合方法
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    
    #选用啥GNN，默认是GIN
    parser.add_argument('--gnn_type', type=str, default="gin")
    
    #数据是哪个，默认tox21，目前只能做分类的
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    
    #可以读取model
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    
    #输出的文件名是什么
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    
    #分割数据集的种子
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    
    #这是个啥seed？
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    
    #划分数据集的方式
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    
    #要不要看training的结果，0应该是不看的
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    
    #这是个啥？加载数据集的工人数量？
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    
    #解析命令行
    args = parser.parse_args()

    #为CPU设置种子用于生成随机数，看起来应该是initialize参数时的种子
    torch.manual_seed(args.runseed)
    #为了数据集划分生成种子
    np.random.seed(args.runseed)
    #设定GPU
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    
    #写了每一个数据集下有多少个任务
    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        #如果没有这个dataset就报这个错
        #那我在上面再多加一条用上我们自己的数据集
        raise ValueError("Invalid dataset name.")

    #set up dataset
    #准备数据集
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
