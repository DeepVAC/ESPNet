
import pickle
import os
import numpy as np
import cv2
import torch.utils.data
from deepvac import LOG

class myDataLoader(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, imList, labelList, transform=None):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        # if you have 255 label in your label files, map it to the background class (19) in the Cityscapes dataset
        if 255 in np.unique(label):
            label[label==255] = 19

        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label)


class LoadData:
    '''
    Class to laod the data
    '''
    def __init__(self, data_dir, classes, cached_data_file, normVal=1.10):
        '''
        :param data_dir: directory where the dataset is kept
        :param classes: number of classes in the dataset
        :param cached_data_file: location where cached file has to be stored
        :param normVal: normalization value, as defined in ERFNet paper
        '''
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.cached_data_file = cached_data_file

    def compute_class_weights(self, histogram):
        '''
        Helper function to compute the class weights
        :param histogram: distribution of class samples
        :return: None, but updates the classWeights variable
        '''
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFile(self, fileName, trainStg=False):
        '''
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        '''
        if trainStg == True:
            global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                line_arr = line.split(',')
                img_file = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()
                label_img = cv2.imread(label_file, 0)
                unique_values = np.unique(label_img)
                # if you have 255 label in your label files, map it to the background class (19) in the Cityscapes dataset
                if 255 in unique_values:
	                label_img[label_img==255] = 19
                
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if trainStg == True:
                    hist = np.histogram(label_img, self.classes)
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)
                    self.mean[0] += np.mean(rgb_img[:,:,0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes {}.'.format(self.classes-1))
                    print('You have following values as class labels:')
                    print(unique_values)
                    print('Some problem with labels. Please check image file: {}'.format(label_file))
                    print('Exiting!!')
                    exit()
                no_files += 1

        if trainStg == True:
            # divide the mean and std values by the sample space size
            self.mean /= no_files
            self.std /= no_files

            #compute the class imbalance information
            self.compute_class_weights(global_hist)
        return 0

    def processData(self, data_path):
        '''
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        print('Processing training data')
        return_val = self.readFile(os.path.join(data_path,'train.txt'), True)

        print('Processing validation data')
        return_val1 = self.readFile(os.path.join(data_path,'val.txt'))

        print('Pickling data')
        if return_val == 0 and return_val1 == 0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList

            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights

            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            return data_dict
        return None

def getData(cached_data_file, class_num, data_path='data'):
    if os.path.isfile(cached_data_file):
        return pickle.load(open(cached_data_file, "rb"))

    dataLoad = LoadData('', class_num, cached_data_file)
    data = dataLoad.processData(data_path)
    if data is None:
        LOG.logE('Error while pickling data. Please check.', exit=True)
    return data

