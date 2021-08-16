'''
Based on
https://github.com/taoxugit/AttnGAN/blob/master/code/pretrain_DAMSM.py
'''

from __future__ import print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.utils import build_super_images2
from miscc.losses import sent_loss, words_loss, image_to_text_loss
from miscc.config import cfg, cfg_from_file

from datasets import ChiTextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER_RNN_DECODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import pickle
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/STREAM/bird.yaml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=50, help='step size for saving trained models')
    parser.add_argument('--img_save_ep_range', dest='img_save_ep_range', type=int, default=10, help='save attention images up to and including specified epoch')
    parser.add_argument('--img_save_ep', dest='img_save_ep', type=int, default=2, help='epoch step size for saving attention images')
    parser.add_argument('--img_save_step', dest='img_save_step', type=int, default=50, help='batch step size for saving attention images')
    args = parser.parse_args()
    return args


def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()

    total_loss = 0
    t_total_loss = 0
    s_total_loss = 0
    w_total_loss = 0
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0

    tot_losses = []
    t_tot_losses = []
    s_tot_losses = []
    w_tot_losses = []
    s_losses0 = []
    s_losses1 = []
    w_losses0 = []
    w_losses1 = []
    ms_per_batch = []

    count = (epoch + 1) * len(dataloader)
    start_time = time.time()

    for step, data in enumerate(dataloader, 0):
        # print('step', step)
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, \
            class_ids, keys = prepare_data(data)

        # sent_code: batch_size x nef
        words_features, sent_code, word_logits = cnn_model(imgs[-1], captions)
        # bs x T x vocab_size

        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
        loss = w_loss0 + w_loss1
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        w_total_loss += (w_loss0.data + w_loss1.data)

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        s_total_loss += (s_loss0.data + s_loss1.data)

        t_loss = image_to_text_loss(word_logits, captions)
        loss += t_loss
        t_total_loss += t_loss.data

        total_loss += ((w_loss0.data + w_loss1.data) + \
                       (s_loss0.data + s_loss1.data) + \
                        t_loss.data)

        loss.backward()

        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if (step + 1) % args.log_step == 0:
            count = epoch * len(dataloader) + step

            avg_loss = total_loss.item() / args.log_step
            t_avg_loss = t_total_loss.item() / args.log_step
            s_avg_loss = s_total_loss.item() / args.log_step
            w_avg_loss = w_total_loss.item() / args.log_step
            s_avg_loss0 = s_total_loss0.item() / args.log_step
            s_avg_loss1 = s_total_loss1.item() / args.log_step
            w_avg_loss0 = w_total_loss0.item() / args.log_step
            w_avg_loss1 = w_total_loss1.item() / args.log_step

            elapsed = time.time() - start_time
            print('Epoch [{}/{}] | Batch [{}/{}] | ms/batch {:5.2f} | Total loss {:5.2f} | Total t_loss {:5.2f} | Total s_loss {:5.2f} | Total w_loss {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, cfg.TRAIN.MAX_EPOCH, step + 1, len(dataloader),
                          elapsed * 1000. / args.log_step, avg_loss, t_avg_loss, s_avg_loss, w_avg_loss,
                          s_avg_loss0, s_avg_loss1,
                          w_avg_loss0, w_avg_loss1))

            tot_losses.append(avg_loss)
            t_tot_losses.append(t_avg_loss)
            s_tot_losses.append(s_avg_loss)
            w_tot_losses.append(w_avg_loss)
            s_losses0.append(s_avg_loss0)
            s_losses1.append(s_avg_loss1)
            w_losses0.append(w_avg_loss0)
            w_losses1.append(w_avg_loss1)
            ms_per_batch.append(elapsed * 1000. / args.log_step) 

            total_loss = 0
            t_total_loss = 0
            s_total_loss = 0
            w_total_loss = 0
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()

        if ((epoch <= args.img_save_ep_range and epoch % args.img_save_ep == 0) or epoch == cfg.TRAIN.MAX_EPOCH) and (step + 1) % args.img_save_step == 0:
            print('Attention map saved: Epoch {} - Step {}'.format(epoch, step + 1))
            # attention Maps
            img_set, _ = \
                build_super_images(imgs[-1].cpu(), captions,
                                   ixtoword, attn_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps_%d_%d.png' % (image_dir, epoch, step + 1)
                im.save(fullpath)

    return count, tot_losses, t_tot_losses, s_tot_losses, w_tot_losses, s_losses0, s_losses1, w_losses0, w_losses1, ms_per_batch


def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()

    total_loss = 0
    t_total_loss = 0
    s_total_loss = 0
    w_total_loss = 0
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0

    for step, data in enumerate(dataloader, 0):
        imgs, captions, cap_lens, \
                class_ids, keys = prepare_data(data)

        words_features, sent_code, word_logits = cnn_model(imgs[-1], captions)
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        w_total_loss += (w_loss0.data + w_loss1.data)

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        s_total_loss += (s_loss0.data + s_loss1.data)

        t_loss = image_to_text_loss(word_logits, captions)
        t_total_loss += t_loss.data
        
        total_loss += ((w_loss0.data + w_loss1.data) + \
                       (s_loss0.data + s_loss1.data) + \
                        t_loss.data)

        if (step + 1) == 50:
            break

    avg_loss = total_loss.item() / (step + 1)
    t_avg_loss = t_total_loss.item() / args.log_step
    s_avg_loss = s_total_loss.item() / (step + 1)
    w_avg_loss = w_total_loss.item() / (step + 1)
    s_avg_loss0 = s_total_loss0.item() / (step + 1)
    s_avg_loss1 = s_total_loss1.item() / (step + 1)
    w_avg_loss0 = w_total_loss0.item() / (step + 1)
    w_avg_loss1 = w_total_loss1.item() / (step + 1)

    return avg_loss, t_avg_loss, s_avg_loss, w_avg_loss, s_avg_loss0, s_avg_loss1, w_avg_loss0, w_avg_loss1


def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER_RNN_DECODER(cfg.TEXT.EMBEDDING_DIM, cfg.CNN_RNN.HIDDEN_DIM,
                                            dataset.n_words, rec_unit=cfg.RNN_TYPE)

    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 1
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/pretrain/pretrain_cycle/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    print('Image resized to: ', imsize)
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = ChiTextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    # print(dataset.ixtoword)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = ChiTextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        loss_dict = {}

        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH + 1):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            print('-' * 89)
            count, tot_losses, t_tot_losses, s_tot_losses, w_tot_losses, s_losses0, s_losses1, w_losses0, w_losses1, ms_per_batch = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir)
            print('-' * 89)
            if len(dataloader_val) > 0:
                val_loss, t_val_loss, s_val_loss, w_val_loss, s_val_loss0, s_val_loss1, w_val_loss0, w_val_loss1 = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size)
                print('End epoch {} | Duration of epoch {}s | lr {:.5f} | Val loss {:5.2f} | Val t_loss {:5.2f} | Val s_loss {:5.2f} | Val w_loss {:5.2f} | '
                      's_loss {:5.2f} {:5.2f} | '
                      'w_loss {:5.2f} {:5.2f}'
                      .format(epoch, time.time() - epoch_start_time, lr, val_loss, t_val_loss, s_val_loss, w_val_loss, s_val_loss0, s_val_loss1, w_val_loss0, w_val_loss1))

            loss_dict[epoch] = {}

            loss_dict[epoch]['tot_losses'] = tot_losses
            loss_dict[epoch]['t_tot_losses'] = t_tot_losses
            loss_dict[epoch]['s_tot_losses'] = s_tot_losses
            loss_dict[epoch]['w_tot_losses'] = w_tot_losses
            loss_dict[epoch]['s_losses0'] = s_losses0
            loss_dict[epoch]['s_losses1'] = s_losses1
            loss_dict[epoch]['w_losses0'] = w_losses0
            loss_dict[epoch]['w_losses1'] = w_losses1

            loss_dict[epoch]['val_loss'] = val_loss
            loss_dict[epoch]['t_val_loss'] = t_val_loss
            loss_dict[epoch]['s_val_loss'] = s_val_loss
            loss_dict[epoch]['w_val_loss'] = w_val_loss
            loss_dict[epoch]['s_val_loss0'] = s_val_loss0
            loss_dict[epoch]['s_val_loss1'] = s_val_loss1
            loss_dict[epoch]['w_val_loss0'] = w_val_loss0
            loss_dict[epoch]['w_val_loss1'] = w_val_loss1

            loss_dict[epoch]['ms_per_batch'] = ms_per_batch
            loss_dict[epoch]['lr'] = lr

            with open(os.path.join(output_dir, 'loss_dict.pickle'), 'wb') as f:
                pickle.dump(loss_dict, f, protocol=2)

            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % args.save_step == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH) or (epoch <= 16 and epoch % 2 == 0):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

