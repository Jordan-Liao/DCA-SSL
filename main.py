import argparse
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import utils.dataset as dataset
import torchvision.models as models
from network.builder import get_model
import utils.loss as loss
import sys
import logging
import datetime
from utils.loss import MMDLoss as mmd

def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging

history_parameters = None

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Pre-Training')
parser.add_argument('--path', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--log_dir', metavar='DIR',default='./log',
                    help='path to save state_dict')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='BIDFC_model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total iteration to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b','--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size for training(default: 512), this is the total '
                         'batch size of all GPUs on the current node when ')
parser.add_argument('--lr', '--learning-rate', default=0.3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--SGD', default=False, nargs='*', type=bool,
                    help='option for optimizer(SGD or Adam)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-s', '--save_freq', default=10, type=int,
                    metavar='N', help='save state_dict frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=10, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--MLClassifier', default="Logistic_Regression", type=str,
                    help='classifier from machine learning; option: KNN,logistic_regression')

# DCA-SSL specific configs:
parser.add_argument('-k','--augment_times', default=6, type=int,
                    help='K times augmentation (default:5)')
parser.add_argument('-T', default=2.0, type=int,
                    help='Temperature Coefficient (default:2.0)')
parser.add_argument('-T1', default=30, type=int,
                    help='A parameter for the weight of DWVLoss')
parser.add_argument('-T2', default=150, type=int,
                    help='A parameter for the weight of DWVLoss')
parser.add_argument('-af',default=1.0,type=int,
                    help='A parameter for the weight of DWVLoss')
parser.add_argument('-ema',default=0.999,type=int,
                    help='A parameter for the BIDFC model update')
parser.add_argument('-alpha',default=0.1, type=float,
                    help='A parameter for the ContrastiveCrop')


def main():
    logging = init_logger(log_dir='log')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    args = parser.parse_args()
    best_acc = 0.0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

        print("Use GPU: {} for training".format(args.gpu))

    # create DCA-SSL_model
    print("=> creating DCA-SSL_model '{}'".format(args.arch))
    model = get_model(
        args.arch,
        args.batch_size).cuda(args.gpu)
    model.cuda(args.gpu)
    print(model)

    # define loss function (criterion) and optimizer
    # d = 0.9
    # m = 0
    PALoss = loss.CrossEntropyLoss().cuda(args.gpu)
    FALoss = loss.FALoss().cuda(args.gpu)
    # Loss = d*loss.FALoss().cuda(args.gpu)+m*loss.Meansar.cuda(args.gpu)
    criterion = [PALoss,FALoss]


    args.lr = args.lr * args.batch_size / 512.
    if args.SGD:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map DCA-SSL_model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    # Config the training loader
    # traindir = os.path.join(args.path, 'train')
    traindir = r"/media/aoc/data/MSTAR/train"
    train_transformation = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(30)], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4)],p=0.4),
            transforms.ToTensor(),
            # transforms.RandomApply([dataset.GaussianNoise()], p=0.2),
            # transforms.RandomApply([dataset.GaussianNoise(random.uniform(0, 0.5))], p=1),
            transforms.RandomApply([Speckle_Noise(random.uniform(0.7, 1))], p=0.8),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                     std=[0.5,0.5,0.5])

            # transforms.Grayscale(3),
            # transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([dataset.New_GaussianBlur(kernel_size=int(0.1 * 64))], p=0.8),
            # transforms.RandomApply([transforms.RandomRotation(30)], p=0.8),
            # transforms.ColorJitter(0.4 , 0.4 , 0, 0),
            # transforms.ToTensor(),
            # transforms.RandomApply([dataset.GaussianNoise()], p=0.8),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
        ]
    )
    train_dataset = dataset.DCA_Dataset(traindir, args.augment_times,
                                         args.batch_size, train_transformation)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        trainer(train_loader, model, criterion, optimizer, epoch, args)
        if (epoch+1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args.log_dir, filename='checkpoint_{:04d}.pth.tar'.format(epoch+1))
            logging.info('checkpoint_{:04d}.pth.tar saved!'.format(epoch+1))


def trainer(train_loader,model,criterion,optimizer,epoch,args):
    global history_parameters
    history_parameters = model.parameters()
    model.train()
    PALoss,FALoss = criterion[0],criterion[1]

    cls_running_loss = 0.0


    for batch,(img_list,target_list) in enumerate(train_loader):#image_list.shape([256,7,3,64,64]) target_list.shape([256,7])
        nn.init.xavier_normal(model.fc.weight)#fc
        img_list = img_list.transpose(1, 0)#image_list.shape([7,256,3,64,64])
        target_list = target_list.transpose(1, 0)#target_list.shape([7,256])

        # K Times Backpropagation for CrossEntropyLoss
        for i in range(args.augment_times):
            permutation = torch.randperm(target_list[i].shape[0])#permutation.shape=256
            data = img_list[i][permutation, :, :, :]#data.shape=256img_list = {Tensor: (7, 256, 3, 64, 64)} tensor([[[[[-6.2388e-01, -1.0000e+00, -1.0000e+00,  ..., -9.7198e-01,\n            -1.0000e+00, -7.8247e-01],\n           [ 4.1247e-01, -1.0000e+00, -6.6143e-01,  ..., -7.3431e-01,\n             3.4255e-01, -9.0022e-01],\n           [-3.0199e-01, -8.4956e-01, … View
            target = target_list[i][permutation]#
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)

            optimizer.zero_grad()
            out = model(data)
            cls_loss = FALoss(out/args.T, target)
            cls_running_loss = cls_loss.item()
            cls_loss.backward()
            optimizer.step()

        # One Time Backpropagation for DWVLoss
        dwv_input = torch.tensor(0).cuda(args.gpu)
        optimizer.zero_grad()
        for i in range(args.augment_times):
            data = img_list[i].cuda(args.gpu)
            model.avgpool.register_forward_hook(get_activation('avgpool'))
            model(data)
            if i == 0:
                dwv_input = torch.unsqueeze(activation['avgpool'], 1)
            else:
                dwv_input = torch.cat([dwv_input, torch.unsqueeze(activation['avgpool'], 1)], dim=1)#torch.Size([256, 3, 512, 1, 1]) 当i=3
        # dwv_input_mean = torch.mean(dwv_input, dim=1, keepdim=True)
        # dwv_loss = FA_Weight(epoch,args.af,args.T1,args.T2) * FALoss(dwv_input, dwv_input_mean)
        dwv_loss = FA_Weight(epoch, args.af, args.T1, args.T2) * FALoss(dwv_input)
        dwv_loss.backward()
        optimizer.step()
        update_ema_variables(model, args.ema)

        print('(epoch: {} / batch: {} / augment_times: {}) cls_loss: {:.6f} dwv_loss: {:.6f}'.format(
            epoch+1, batch + 1, args.augment_times, cls_running_loss, dwv_loss.item()))

        logging.debug('(epoch: {} / batch: {} / augment_times: {}) cls_loss: {:.6f} dwv_loss: {:.6f}'.format(
            epoch+1, batch + 1, args.augment_times, cls_running_loss, dwv_loss.item()))


activation = {}

class Speckle_Noise(object):
    """Speckle Noise Augmentation for tensor"""

    def __init__(self, sigma=1.5):
        self.sigma = sigma

    def __call__(self, img):
        noise = torch.randn(1,img.shape[1],img.shape[2])#[1,64,64]
        noise = torch.cat([noise,noise,noise],dim=0)#[3,64,64]
        speck = torch.exp(noise)
        # speck = torch.min(torch.exp(noise),self.sigma)
        speck[speck > self.sigma]= self.sigma
        img = img*speck
        img[img > 1] = 1
        img[img < 0] = 0
        return img

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def FA_Weight(epoch,af,T1,T2):
    alpha = 0.0
    if epoch > T1:
        alpha = (epoch-T1) / (T2-T1)*af
        if epoch > T2:
            alpha = af
    return alpha


def update_ema_variables(model, alpha):
    # Use the true average until the exponential average is more correct
    global history_parameters
    for history_param, param in zip(history_parameters, model.parameters()):
        param.data = alpha * history_param.data + (1-alpha) * param.data
    history_parameters = model.parameters()


def save_checkpoint(state, log_dir, filename='checkpoint.pth.tar'):
    filename_path = os.path.join(log_dir, filename)
    torch.save(state, filename_path)
    print(filename + " has been saved.")


if __name__ == '__main__':
    main()