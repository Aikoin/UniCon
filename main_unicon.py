from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from losses import SupConLoss, UniConLoss
from networks.resnet_big import ConResNet
from test import test, get_test_features, get_train_features, set_classifier
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import get_universum
from util import set_optimizer, save_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--test_freq', type=int, default=10,
                        help='test frequency')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='140,160,180',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # mixup
    parser.add_argument('--lamda', type=float, default=0.5, help='universum lambda')
    parser.add_argument('--mix', type=str, default='mixup', choices=['mixup', 'cutmix'], help='use mixup or cutmix')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'imagenet'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')


    # method
    parser.add_argument('--method', type=str, default='UniCon', choices=['UniCon', 'SupCon', 'SimCLR'],
                        help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    dic = {'cifar10':10, 'cifar100':100, 'tinyimagenet':200, 'imagenet':1000}
    opt.n_classes = dic[opt.dataset]

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/{}_models/{}'.format(opt.dataset, opt.method)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_{}_lambda_{}_trial_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.mix, opt.lamda, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.1
        opt.warm_epochs = 2
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tinyimagenet' or opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder, train=True,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        test_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, transform=transforms.Compose([
            transforms.Resize((opt.size, opt.size)), transforms.ToTensor(), normalize, ]), download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder, train=True,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        test_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, transform=transforms.Compose([
            transforms.Resize((opt.size, opt.size)), transforms.ToTensor(), normalize, ]), download=True)
    elif opt.dataset == 'tinyimagenet':
        train_dataset = datasets.ImageFolder(root=opt.data_folder + 'tiny-imagenet-200/train/',
                                             transform=TwoCropTransform(train_transform))
        test_dataset = datasets.ImageFolder(root=opt.data_folder + 'tiny-imagenet-200/val/',
                                            transform=transforms.Compose([
                                                transforms.Resize((opt.size, opt.size)), transforms.ToTensor(),
                                                normalize, ]))
    elif opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=opt.data_folder + 'train/',
                                             transform=TwoCropTransform(train_transform))
        test_dataset = datasets.ImageFolder(root=opt.data_folder + 'val/',
                                            transform=transforms.Compose([
                                                transforms.Resize((opt.size, opt.size)), transforms.ToTensor(),
                                                normalize, ]))
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder+'train/',
                                             transform=TwoCropTransform(train_transform))
        test_dataset = datasets.ImageFolder(root=opt.data_folder+'val/',
                                            transform=transforms.Compose([
                                                transforms.Resize((opt.size, opt.size)), transforms.ToTensor(),
                                                normalize, ]))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    return train_loader, test_loader


def set_model(opt):
    torch.cuda.empty_cache()

    model = ConResNet(name=opt.model)

    if opt.method == 'UniCon':
        criterion = UniConLoss(temperature=opt.temp)
    else:
        criterion = SupConLoss(temperature=opt.temp)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            '''单卡保存，多卡加载'''
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     if k.startswith('encoder.'):
            #         k = k.replace('encoder.', 'encoder.module.')    
            #     new_state_dict[k] = v
            # state_dict = new_state_dict
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            '''多卡保存，单卡加载'''
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        model.load_state_dict(state_dict, strict=True)
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        # images: a list of length 2，each element being a tensor of size [128, 3, 32, 32]
        # labels: vector of length 128

        data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        if opt.method == 'UniCon':
            # get universum
            universum = get_universum(images, labels, opt)
            uni_features = model(universum)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if opt.method == 'UniCon':
            loss = criterion(features, uni_features, labels)
        elif opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():

    opt = parse_option()
    print(opt)

    # build data loader
    train_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    best_acc = 0
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # test the model
        if epoch % opt.test_freq == 0:
            features, y = get_train_features(train_loader, model)
            classifier = set_classifier(features, y, opt)
            features, y = get_test_features(test_loader, model)
            acc = test(features, y, classifier)
            if acc > best_acc:
                best_acc = acc
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))    


if __name__ == '__main__':
    main()
