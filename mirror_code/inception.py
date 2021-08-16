import os
import torch
import argparse
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from torch.utils.data.dataset import Dataset

from skimage import io


class GeneratedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.all_images = []
        for dir in list(os.listdir(self.root_dir)):
            if dir == '.DS_Store': continue
            if 'birds' not in self.root_dir:
                self.all_images.append(dir)
            else:
                for filename in list(os.listdir(os.path.join(self.root_dir, dir))):
                    self.all_images.append(os.path.join(dir, filename))


    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.all_images[idx])
        image = io.imread(img_name)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    print('Amount of images: {}'.format(len(imgs)))

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    print('Loaded Inception-v3 model')
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        print('Batch {}/{}'.format(i, len(dataloader)))
        batch = batch.transpose(1, 3)

        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='validation images directory',
                        default='', type=str)
    parser.add_argument('--gpu', dest='gpu', type=bool, default=True)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--resize', dest='resize', type=bool, default=False)
    parser.add_argument('--splits', dest='splits', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    imgs = GeneratedDataset(args.img_dir)

    print(inception_score(imgs, args.gpu, args.batch_size, args.resize, args.splits))
