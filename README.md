# Face mask detection using CNN

This work aims to demonstrate an end–to–end pipeline 
implementation of a Convolutional Neural Network (CNN–based) Face Mask detector to
classify five categories of face masks, namely:
* A person without a face mask.
* A person with a “cloth” face mask.
* A person with a “surgical” mask.
* A person with a “FFP2/N95/KN95”-type mask.
* A person with a mask worn incorrectly.


## Getting Started

These instructions will get you a copy of the project up and running on your 
local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
*  The model architecture can be switched inside the CNN.py file
*  The Jupiter notebook can be used to test the pre-trained model on a downloaded picture.

### Prerequisites

* Python 3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
* Pytorch
* StreamLit
* Anaconda packages for jupyter notebook

```
pip install python
```

### Installing

A step by step series of examples that tell you how to get a development env running

* Streamlit
* pytorch
* Jupyter notebook

```
pip install stremlit
pip install torchvision
```

```
pip install pytorch
```
### To test a pre-trained model 
A quick start script to test our pre-trained model can be found in the notebook file

## Running the tests

* clone the repository on your PC
* [Download dataset and put in folder Data] (https://drive.google.com/drive/folders/14o0j_ww9-E0rnHzCnpwJ7L7VNNKQn91p?usp=sharing/)
* The model architecture can be switched inside the CNN.py file
* The Jupiter notebook can be used to test the pre-trained model on a downloaded picture.

## Part 2 Evaluation using Bias and K-fold for each categories
* Link to unbiased dataset for part 2: https://drive.google.com/file/d/1qutuBEG_r-AMu6SzKmzVkJKyzeYZUykh/view?usp=sharing
* There are two additional models used to improve evaluations
* 1- The gender-based unbiased model that uses the Data_GENDER folder from dataset 2 for evaluation and
* 2- The age-based unbiased model that uses the Data_AGE folder for evaluation each improve performance metric by limiting overfitting
* It is important to change the path of data folder in preprocessing.py for one of the two also set the model number to use for the 10 fold evaluation (see preporcessing.py e.g path = os.getcwd() + '/Data_AGE').
* Set model number and path to pretrained model if necessary

### Break down into end to end tests

To train the model please run the following to generate the trained model. There are 3 models to choose from, you can change the model number in the CNN.py  
* [1] Deep-fewer-maxpooling
* [2] Shallow-network
* [3] Deep-many-maxpooling

```
python cnn.py
```

### To test the model in the app use the following command

```
streamlit run app.py
```

## Deployment

The system application is deployed in Streamlit 

## Built With

* [Streamlit](https://streamlit.io/) - The web framework used
* [Python](https://www.python.org/) - Dependency Management and programming
* [Notebook Jupyter](https://jupyter.org/) - Produce useful results and comparisons

## CNN Model

## Versioning

We use Github for versioning. For the versions available, see the [tags on this repository](https://github.com/hormone03/AI_project). 
The report was generate in LaTex

## Authors

* **Akinlolu Ojo**
* **Herve Ngomseu Fotsing** 
* **Isaac Oluwole Olukunle**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Concordia University
* Participating team mates
* Professors of the course
