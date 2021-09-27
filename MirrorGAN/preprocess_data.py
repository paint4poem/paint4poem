import csv
import os
import numpy as np
import re
import pickle
import argparse
from nltk.tokenize import RegexpTokenizer
import random

def title_image_prep(data_dir, csv_name):
    text_path = os.path.join(data_dir, 'text')
    if not os.path.isdir(text_path):
        os.makedirs(text_path)

    with open(os.path.join(data_dir, csv_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        files = []
        cnt_empty = 0
        cnt_english = 0
        for index, row in enumerate(csv_reader):
            file = row[12]
            cap = row[5]
            # check if string contains chinese characters
            if re.search(u'[\u4e00-\u9fff]', cap):
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                tokens = [j for i in tokens for j in i]
                # check if tokens contain more than one english character
                if sum([t.isascii() for t in tokens]) > 1:
                    cnt_english += 1
                    print(index, tokens)
                else:
                    files.append(file)
                    with open(text_path + '/' + file + '.txt', "w") as txt:
                        txt.write(cap)
            elif cap == '':
                cnt_empty += 1
            else:
                cnt_english += 1

    print('total amount english (delete this from data): {}'.format(cnt_english))
    print('total amount empty (delete this from data): {}'.format(cnt_empty))

    return files

def main(args):
    train_path = os.path.join(args.data_dir, 'train')
    if not os.path.isdir(train_path):
        os.makedirs(train_path)

    test_path = os.path.join(args.data_dir, 'test')
    if not os.path.isdir(test_path):
        os.makedirs(test_path)

    if args.data_dir == 'data/title_image':
        filenames = title_image_prep(args.data_dir, 'TITLE-IMAGE.csv')
    elif args.data_dir == 'data/poem_image':
        filenames = title_image_prep(args.data_dir, 'POEM-IMAGE.csv')
    else:
        filenames = os.listdir(os.path.join(args.data_dir, 'text'))
        filenames = [name.rpartition('.txt')[0] for name in filenames]

    print('total amount of data: {}'.format(len(filenames)))

    np.random.shuffle(filenames)
    split_idx = int(0.75 * len(filenames))
    train_files, test_files = filenames[:split_idx], filenames[split_idx:]
    print('total amount of training data: {}'.format(len(train_files)))
    print('total amount of test data: {}'.format(len(test_files)))

    with open(os.path.join(train_path, 'filenames.pickle'), 'wb') as f:
        pickle.dump(train_files, f, protocol=2)

    with open(os.path.join(test_path, 'filenames.pickle'), 'wb') as f:
        pickle.dump(test_files, f, protocol=2)

    with open(os.path.join(test_path, 'filenames.pickle'), 'rb') as f:
        test = pickle.load(f)
        print('10 test filenames:')
        print(test[:10])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    args = parser.parse_args()
    main(args)