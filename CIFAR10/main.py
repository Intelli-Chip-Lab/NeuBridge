import argparse
import os
import sys
import time
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
 
from models import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res20')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('--log', default='output', type=str, help='log document')

parser.add_argument('--epoch_tau_train', default=0, type=int, metavar='N', help='start epoch to train tau')
parser.add_argument('--tau_init', default='2', type=str, help='initial value of tau')
parser.add_argument('--alpha_init', default='2', type=str, help='initial value of alpha')
best_prec = 0
args = parser.parse_args()
torch.cuda.set_device(int(args.device))


#### matching tau_trans between layers ####
def tau_match(snn):
    tau_list = [torch.tensor(3.0)]
    for m in snn.modules():
        if isinstance(m, IF):
            m.act_tau_trans = tau_list[-1]
            tau_list.append(m.act_tau.data)
        elif isinstance(m, last_Spiking):
            m.tau_trans = tau_list[-1]

#### log file ####
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():

    global args, best_prec
    if not os.path.exists(os.path.dirname(__file__)+'/logs'):
        os.makedirs(os.path.dirname(__file__)+'/logs')
    log_document = os.path.dirname(__file__)+'/logs/'+args.log+'.log'
    sys.stdout = Logger(log_document, sys.stdout)
    use_gpu = torch.cuda.is_available()
    print(args.device)
    print('=> Building model...')
    model=None
    if use_gpu:
        float = True if args.bit == 32 else False
        if args.arch == 'alex':
            model = AlexNet(float=float, bit = args.bit)
            snn = S_AlexNet(T = args.bit)
        elif args.arch == 'vgg11':
            model = VGG11(float=float)
            if not float:
                snn = S_VGG11(T = args.bit)

        elif args.arch == 'vgg16_cifar':
            model = VGG16_cifar(float=float)
            if not float:
                snn = S_VGG16_cifar(T = args.bit)
        else:
            print('Architecture not support!')
            return
        if not float:
            for m in model.modules():
                #Ouroboros-------determine quantization
                #APoT quantization for weights, uniform quantization for activations
                if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                    #weight quantization, use APoT
                    m.weight_quant = weight_quantize_fn(w_bit=args.bit, power=True)
                if isinstance(m, QuantReLU):
                    #activation quantization, use uniform
                    # m.act_grid = build_power_value(args.bit)
                    # m.act_alq = act_quantization(b=args.bit, grid=m.act_grid, power=False)
                    m.act_alq = act_quantization_AP(args.bit)
                    m.act_tau = torch.nn.Parameter(torch.tensor(str_to_float(args.tau_init)))
                    m.act_alpha = torch.nn.Parameter(torch.tensor(str_to_float(args.alpha_init)))
            for m in snn.modules():
                #Ouroboros-------determine quantization
                #APoT quantization for weights, uniform quantization for activations
                if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                    #weight quantization, use APoT
                    m.weight_quant = weight_quantize_fn(w_bit=args.bit, power=True)
                # if isinstance(m, QuantReLU):
                #     #activation quantization, use uniform
                #     m.act_grid = build_power_value(args.bit)
                #     m.act_alq = act_quantization(b=args.bit, grid=m.act_grid, power=False)          

        model = model.cuda()
        # model = nn.DataParallel(model).cuda()
        if not float:
            snn = snn.cuda()
            # snn = nn.DataParallel(snn).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/'+str(args.arch)+'_'+str(args.bit)+'bit'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.init, map_location='cpu')
            
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            # for m in model.modules():
            #     if isinstance(m, QuantReLU):
            #         m.act_tau.data = torch.tensor(str_to_float(args.tau_init))
            #         m.act_alpha.data = torch.tensor(str_to_float(args.alpha_init))
            model.show_params()
            if not float:
                snn.load_state_dict(checkpoint['state_dict'],strict=False)
                tau_match(snn)
        else:
            print('No pre-trained model found !')
            exit()


    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch']
                global best_prec
                best_prec = checkpoint['best_prec']
                model.load_state_dict(checkpoint['state_dict'])
                # model.show_params()
                if not float:
                    snn.load_state_dict(checkpoint['state_dict'])
                    tau_match(snn)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    print('=> loading cifar10 data...')
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])  # 定义归一化转换
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,  
        download=True, 
        transform=transforms.Compose([ 
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), 
            CIFAR10Policy(),
            transforms.ToTensor(), 
            Cutout(n_holes=3, length=2),
            normalize, 
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(testloader, model, criterion)
        if not float:
            snn.load_state_dict(model.state_dict())
            tau_match(snn)
            validate(testloader, snn, criterion)
        model.show_params()
        # model.module.show_params()
        # snn.module.show_params()
        return
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # scheduler.step()

        if epoch == args.epoch_tau_train:
            if not float: 
                for m in model.modules():  
                    if isinstance(m, QuantReLU):  
                        m.tau_train = True

        # train for one epoch
        if epoch%10 == 1:
            model.show_params()
            # model.module.show_params()
            if not float:
                snn.show_params()
                # snn.module.show_params()
        train(trainloader, model, criterion, optimizer, epoch)
        tau_list = []
        alpha_list = []
        for m in model.modules():  
            if isinstance(m, QuantReLU): 
                tau_list.append(m.act_tau)
                alpha_list.append(m.act_alpha)
        print('tau_list: {}'.format(torch.tensor(tau_list).tolist()))
        print('alpha_list: {}'.format(torch.tensor(alpha_list).tolist()))

        # evaluate on test set
        prec = validate(testloader, model, criterion)
        if not float:
            snn.load_state_dict(model.state_dict())
            tau_match(snn)
            prec1 = validate(testloader, snn, criterion)
            if prec1 > prec:
                prec = prec1
        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        print('best acc: {:1f}'.format(best_prec))
        save_checkpoint({
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    start = time.time()
    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            
    end = time.time()
    print(f'time {end-start:3f}')

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start = time.time()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}%  time '.format(top1=top1))

    end = time.time()
    print(f'{end-start:3f}')

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

def str_to_float(value):
    return float(value)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__=='__main__':
    main()
