# simple_linear_regression_model
This project offers you a linear regression model in python, a great statistical model for predictions.

## What's the use ?
* Overoming a lacking dataset by completing it.
* Predicting data out of our range of time.
 
## How does it work ?
* The model is trained to recognize the logical progression of a dataset and therefore giving predictions on data it has never came accross.
In other words, the model doesn't "memorize" but "generalize" the data it has learnt.

## Usage case
* Predict the price of a car based on its mileage.
* Predict the height of a specfic plant based on its age.
* Predict the cancer cell count of a patient based on the timeframe of the cancer.

## Getting started

### Installation
* Install the repository in your desired folder.
```
git clone git@github.com:Lucasdye/simple_linear_regression.git
```
* Install the required packages.
```
pip install -r requierements.txt
```
## Usage
* In the 'dataset' folder at the root of the repository, replace the 'dataset_example.csv' by your own desired dataset.
* In the file linear_regression/params.py, replace 'dataset_example.csv' at the end of the 'dataset_path' attribut by the name of your own dataset.
* Still in the file linear_regression/params.py, define the 'lr' (a standard learning rate is between 0 and 1) and 'epoch' attributs.
* Train the model.
  ```
  python3 model_training.py
  ```
* Now you can enter a value input (int or float) for a prediction.
  ```
  python3 model_prediction.py
  ```
## Important Information
This is a supervised training model ! You'll have to play with the learning rate and epoch values for the model to get the right understanding of the dataset.










