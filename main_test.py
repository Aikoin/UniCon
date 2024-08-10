import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from losses import SupConLoss, UniConLoss
from networks.resnet_big import ConResNet
from test import test, get_test_features, get_train_features, set_classifier
from util import TwoCropTransform


def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')

    # optimization for classifier
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'imagenet'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')

    # other setting
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    dic = {'cifar10':10, 'cifar100':100, 'tinyimagenet':200, 'imagenet':1000}
    opt.n_classes = dic[opt.dataset]

    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    
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
    train_transform = TwoCropTransform(train_transform)

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize,
    ])
 
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'tinyimagenet':
        train_dataset = datasets.ImageFolder(root=opt.data_folder + 'tiny-imagenet-200/train/',
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder + 'tiny-imagenet-200/val/',
                                           transform=val_transform)
    elif opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=opt.data_folder + 'train/',
                                             transform=TwoCropTransform(train_transform))
        val_dataset = datasets.ImageFolder(root=opt.data_folder + 'val/',
                                            transform=transforms.Compose([
                                                transforms.Resize((opt.size, opt.size)), transforms.ToTensor(),
                                                normalize, ]))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    return train_loader, test_loader


def set_model(opt):
    torch.cuda.empty_cache()

    model = ConResNet(name=opt.model)

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
                k = k.replace("module.", "").replace("downsample", "shortcut")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        model.load_state_dict(state_dict, strict=True)
        cudnn.benchmark = True

    return model


def main():
    opt = parse_option()
    print(opt)

    train_loader, test_loader = set_loader(opt)
    model = set_model(opt)

    features, y = get_train_features(train_loader, model)
    classifier = set_classifier(features, y, opt)
    features, y = get_test_features(test_loader, model)
    test(features, y, classifier)

if __name__ == '__main__':
    main()