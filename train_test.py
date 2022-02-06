from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ScanObjectNN
from model import PointNet, DGCNN, GBNet
import numpy as np
from torch.utils.data import DataLoader
from utils import cal_loss
import sklearn.metrics as metrics
from my_dataset import My_Dataset

def train(args, io):
    # create dataset
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'ScanObjectNN':
        train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'my_shapenet':
        dataset = My_Dataset(split='train',npoints=1024)
        test_dataset = My_Dataset(split='test',npoints=1024,data_augmentation=False)
        train_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True)
        test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=False)
    else:
        raise Exception("Dataset Not supported")

    device = torch.device("cuda" if args.cuda else "cpu")

    # load models
    if args.model == 'pointnet':
        if args.dataset == 'modelnet40':
            model = PointNet(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = PointNet(args, output_channels=15).to(device)
        elif args.dataset == 'my_shapenet':
            model = PointNet(args, output_channels=5).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'dgcnn':
        if args.dataset == 'modelnet40':
            model = DGCNN(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = DGCNN(args, output_channels=15).to(device)
        elif args.dataset == 'my_shapenet':
            model = DGCNN(args, output_channels=5).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'gbnet':
        if args.dataset == 'modelnet40':
            model = GBNet(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = GBNet(args, output_channels=15).to(device)
        elif args.dataset == 'my_shapenet':
            model = GBNet(args, output_channels=5).to(device)
        else:
            raise Exception("Dataset Not supported")
    else:
        raise Exception("Not implemented")
    # print(str(model))

    model = nn.DataParallel(model)
    print("\nUsing", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD\n")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam\n")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss

    # train
    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        # Test
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        # update saved model
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            outstr = 'Current Best: %.6f' % best_test_acc
            io.cprint(outstr)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    if args.dataset == 'modelnet40':
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'ScanObjectNN':
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'my_shapenet':
        test_dataset = My_Dataset(split='test',npoints=1024,data_augmentation=False)
        test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0)
    else:
        raise Exception("Dataset Not supported")

    device = torch.device("cuda" if args.cuda else "cpu")

    # load models
    if args.model == 'pointnet':
        if args.dataset == 'modelnet40':
            model = PointNet(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = PointNet(args, output_channels=15).to(device)
        elif args.dataset == 'my_shapenet':
            model = PointNet(args, output_channels=5).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'dgcnn':
        if args.dataset == 'modelnet40':
            model = DGCNN(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = DGCNN(args, output_channels=15).to(device)
        elif args.dataset == 'my_shapenet':
            model = DGCNN(args, output_channels=5).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'gbnet':
        if args.dataset == 'modelnet40':
            model = GBNet(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = GBNet(args, output_channels=15).to(device)
        elif args.dataset == 'my_shapenet':
            model = GBNet(args, output_channels=5).to(device)
        else:
            raise Exception("Dataset Not supported")
    else:
        raise Exception("Not implemented")
    # print(str(model))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
