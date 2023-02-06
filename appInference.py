import torch
import numpy as np
import pandas as pd
from preProcessing import *
from CNN import *
import torchvision.transforms as T
from PIL import Image

labels = ['using cloth mask', 'using surgical mask', 'using FFP2 mask', 'not using mask', 'using mask incorrectly']


def processImage(image):
    # Cal. image parameters (mean and std)
    img = image.convert('L').resize((img_size, img_size))
    img = np.array(img)
    img = img.astype(np.float64) / 255.0
    data_sum = np.sum(img)
    mean = data_sum / (img_size * img_size)
    std = np.std(img)

    # Transform
    image_transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize([mean], [std]),
    ])

    image = image_transforms(image)

    return image


# image_ = processImage(imagePath)

def Test(modelPath, image):
    # Set the CNN module in evaluation mode:
    image = processImage(image)
    image = image.reshape(1, 1, 64, 64)
    model = torch.load(modelPath)
    model.eval()
    # print(summary(cnn, input_size=(1, 64,64)))
    # disable the gradient calculation in test mode
    with torch.no_grad():
        output = model(image)
        max_element, max_idx = torch.max(output.data, 1)
        # print(output)
        print("The person is ", labels[max_idx])

    return output, max_element, max_idx
