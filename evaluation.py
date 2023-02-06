import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from preProcessing import *
from CNN import *

# change depending on model you want to use, also change the Data source line 12
# of preprocessing
model_no_ = 2
top_path = os.getcwd()


def evaluate(cnnModel, test_loader, my_categories):
    # Set the CNN module in evaluation mode:
    cnnModel.eval()

    # print(summary(cnn, input_size=(1, 64,64)))
    # disable the gradient calculation in test mode
    with torch.no_grad():
        classifiedCorrectly = 0
        totalImages = 0
        for images, labels in test_loader:
            outputs = cnnModel(images)
            max_elements, max_idxs = torch.max(outputs.data, 1)
            totalImages += labels.size(0)
            classifiedCorrectly += (max_idxs == labels).sum().item()

    print('\nTest Accuracy of the model on the {} test images: {} %'
          .format((totalImages),
                  (classifiedCorrectly / totalImages) * 100))
    test_labels = np.asarray(labels, dtype=np.float32)
    pre_labels = np.asarray(max_idxs, dtype=np.float32)

    conf_Mat = confusion_matrix(test_labels, pre_labels)
    print('\n Confusion Matrix\n')
    print(conf_Mat, '\n')

    df = pd.DataFrame(conf_Mat, index=my_categories, columns=my_categories)
    print(df, '\n\n')
    print(classification_report(test_labels, pre_labels, target_names=my_categories))

    test_accuracy = (classifiedCorrectly / totalImages) * 100
    return test_accuracy


def CNN_load_trained_model(model_no_):
    dataset, my_categories = pre_processing()

    # print("length of dataset = ", len(dataset))
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_ratio, stratify=dataset.targets,
                                           random_state=0)
    #train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)
    #
    # print("length of training set = ", len(train))
    print("length of testing set = ", len(test))
    #
    # load datas
    # train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False, num_workers=2)
    if model_no_ == 1:
        model = torch.load(top_path + '/trained_model_k_fold.pth')
    elif model_no_ == 2:
        model = torch.load(top_path + '/trained_model_k_fold_gender.pth')
    else:
        model = torch.load(top_path + '/trained_model_k_fold_age.pth')
    evaluate(model, test_loader, my_categories)


if __name__ == '__main__':
    CNN_load_trained_model(model_no_)
