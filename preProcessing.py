import imp
import os
import torchvision.transforms as T
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import numpy as np

top_path = os.getcwd()
path = os.getcwd() + '/Data_GENDER'  # use /Data_GENDER if you are evaluating Gender
img_size = 64
test_ratio = 0.20

''' 
A method for pre-processing images by resizing, converting to grayscale, normalization 
and splitting dataset images into training dataset and testing dataset
'''


def pre_processing():
    # Get Mean and STD
    mean, std, my_categories = cal_mean_std()

    # Pre_Processing
    image_transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize([mean], [std]),
    ])

    # Get Images
    dataset = ImageFolder(path, transform=image_transforms)

    # # splitting dataset, Each category have same number of images
    # print("length of dataset = ", len(dataset))
    # train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_ratio, stratify=dataset.targets,
    #                                        random_state=0)
    # train = Subset(dataset, train_idx)
    # test = Subset(dataset, test_idx)
    #
    # print("length of training set = ", len(train))
    # print("length of testing set = ", len(test))
    #
    # # load datas
    # train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False, num_workers=2)

    return dataset, my_categories


'''
A method to calculate the mean and the standard deviation for the dataset,
to be used in the normalization process
'''


def cal_mean_std():
    # get categories
    my_categories = os.listdir(os.path.basename(path))
    # print(my_categories)
    if '.DS_Store' in my_categories:
        my_categories.remove('.DS_Store')
    # print(my_categories)
    cnt = 0
    data_sum = 0
    data_std_sum = 0
    for category in my_categories:
        # get directory for each category
        sub_path = path + '/' + category
        if os.path.isdir(sub_path):
            flist = os.listdir(sub_path)
            if '.DS_Store' in flist:
                flist.remove('.DS_Store')
            os.chdir(sub_path)
            for image in flist:
                img = Image.open(image).convert('L').resize((img_size, img_size))
                img = np.array(img)
                img = img.astype(np.float64) / 255.0
                data_sum += np.sum(img)
                data_std_sum += np.std(img)
                cnt = cnt + 1
    os.chdir(top_path)
    data_mean = data_sum / (cnt * img_size * img_size)
    data_std = data_std_sum / cnt
    return data_mean, data_std, my_categories
