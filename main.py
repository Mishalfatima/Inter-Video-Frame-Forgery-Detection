from __future__ import unicode_literals, print_function, division
import argparse
import timeit
from datetime import datetime
import socket
import os
import glob
from read_data import *

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from C3D import *

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train_C3D(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print("Device being used:", device)

    save_dir_root = "./"
    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_**')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    print(run_id)
    #run_id = 48
    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

    saveName = args.modelName + '-' + args.dataset

    model = C3D_model(num_classes=2).to(device)

    train_params = [{'params': get_1x_lr_params(model), 'lr': args.lr},
                    {'params': get_10x_lr_params(model), 'lr': args.lr * 10}]

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(train_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)
    if args.resume_epoch == 0:
        print("Training {} from scratch...".format('C3D'))

    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(args.dataset))

    train = Dashcam_data(train='train')

    step = 0
    for epoch in range(args.resume_epoch, args.epochs):
        # each epoch has a training and validation step
        for phase in ['train']:

            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0


            # scheduler.step() is to be called once every epoch during training
            dataset = train
            scheduler.step()
            model.train()

            im_names = (dataset.total_folders)
            tot_batches = int(im_names/args.batch_size)


            for i in range(tot_batches):


                    inputs, labels = dataset.get_next_batch(args.batch_size,args.clip_len)

                    # move inputs and labels to the device the training is taking place on
                    inputs = Variable(inputs).to(device)
                    labels = Variable(labels).to(device)
                    optimizer.zero_grad()

                    k = inputs
                    for j in range(args.clip_len-1):
                        k[:,:,j,:,:] = inputs[:,:,j+1,:,:]-inputs[:,:,j,:,:]


                    loss1 = torch.mean(torch.mean(torch.mean(torch.mean(torch.mean(k,dim = 1),dim = 1),dim= 1),dim= 1),dim=0)

                    outputs = model(inputs)
                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]

                    loss = criterion(outputs, labels)
                    loss = loss+loss1

                    print(loss.item())
                    if (i % 10)==0:
                        print("Epoch", epoch, "Batch done ", i, "out of", tot_batches)
                        print("loss is ", loss.item())
                        writer.add_scalar('data/train_loss_batch', loss.item(), step)
                        step += 1

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = (running_loss / im_names)
            epoch_acc = (running_corrects.item() / im_names)*100
            if phase == 'train':
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, args.epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

            if epoch % args.snapshot == (args.snapshot - 1):
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join('./models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
                print("Save model at {}\n".format(
                    os.path.join('./models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))


    torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
        }, os.path.join('./models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
    print("Save model at {}\n".format(
            os.path.join('./models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))
    writer.close()


def test_C3D(args):

    TP=0
    FN=0
    FP=0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = C3D_model(num_classes=2).to(device)
    checkpoint = torch.load("./models/C3D-dashcam_epoch-24.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    start_time = timeit.default_timer()
    criterion = nn.CrossEntropyLoss().to(device)

    running_loss = 0.0
    running_corrects = 0.0
    dataset = Dashcam_data(train='test')

    im_names = (dataset.total_folders)

    # for inputs, labels in tqdm(test_dataloader):
    print(im_names)
    tot_batches = int(im_names / args.batch_size)
    for i in range(tot_batches):
        print(i, "out of", tot_batches)
        inputs, labels = dataset.get_next_batch(args.batch_size, args.clip_len)

        inputs = inputs.to(device)
        labels = labels.to(device)

        k = inputs
        for j in range(args.clip_len - 1):
            k[:, :, j, :, :] = inputs[:, :, j + 1, :, :] - inputs[:, :, j, :, :]

        loss1 = torch.mean(torch.mean(torch.mean(torch.mean(torch.mean(k, dim=1), dim=1), dim=1), dim=1), dim=0)

        with torch.no_grad():
            outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)
        loss = loss+loss1

        running_loss += loss.item() * inputs.size(0)
        print("preds",preds)
        print("labels.data",labels.data)

        for i in range(args.batch_size):
            if (preds[i]==1 and labels[i] ==1):
                TP+=1
            elif (preds[i]==0 and labels[i] ==1):
                FN+=1

            elif(preds[i]==1 and labels[i]==0):
                FP+=1

        running_corrects += torch.sum(preds == labels.data)

    Recall = TP/(TP+FN)
    print("Recall is ", Recall)

    Precision = TP / (TP + FP)
    print("Precision is ", Precision)

    epoch_loss = running_loss / im_names
    epoch_acc = (running_corrects.item() / im_names)*100

    print("[test] Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, help='epoch number', default=60)
    argparser.add_argument('--Train', type=bool, default=True)
    argparser.add_argument('--continue_training', type=bool, default=True)
    argparser.add_argument('--model', type=str, default="checkpoint")
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--batch_size', type=int, default=2)
    argparser.add_argument('--clip_len', type=int, default=16)
    argparser.add_argument('--resume_epoch', type=int, default=0)
    argparser.add_argument('--dataset', type=str, default="dashcam")
    argparser.add_argument('--save_dir', type=str, default="logs")
    argparser.add_argument('--save_epoch', type=int, default=10)
    argparser.add_argument('--snapshot', type=int, default=1)
    argparser.add_argument('--num_classes', type=int, default=2)
    argparser.add_argument('--modelName', type=str, default="C3D")
    argparser.add_argument('--useTest', type=bool, default=False)
    argparser.add_argument('--nTestInterval', type=int, default=2)
    args = argparser.parse_args()


    if args.Train:

        train_C3D(args)

    else:
        test_C3D(args)


if __name__=='__main__':
    main()