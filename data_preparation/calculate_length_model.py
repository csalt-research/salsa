import torch
import argparse
from argparse import Namespace

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=Path, default="out/adapter/alpaca", help='Path to directory hosting experiment logs and checkpoints') 
    parser.add_argument('--train-data-path', type=Path, default="data/alpaca/train.pt", help='Path to the .pt file containing training data') 
    parser.add_argument('--val-data-path', type=Path, default="data/alpaca/val.pt", help='Path to the .pt file containing validation data')
    return parser

def setup():
    # Parse arguements
    parser = get_parser()
    args = parser.parse_args()

    # Load datasets
    train_data = torch.load(args.train_data_path, map_location=torch.device('cpu'))
    val_data = torch.load(args.val_data_path, map_location=torch.device('cpu'))

    return args, train_data, val_data

def get_data(torch_data):
    input_lengths = []
    output_lengths = []

    for index in range(len(torch_data)):

        data = torch_data[index]

        audio_duration = data["audio_duration"]
        labels = data["labels"]

        input_lengths.append(audio_duration)
        output_lengths.append(labels.shape[0]) 

    return input_lengths, output_lengths

def main():
    ##############################    Setup Training    ################################
    args, train_data, val_data = setup()

    input_lengths, output_lengths = get_data(train_data)

    
    input_lengths = np.array(input_lengths).reshape(-1, 1)
    output_lengths = np.array(output_lengths)

    # Define the degree of the polynomial
    degree = 3  # Change this to the desired degree of the polynomial

    # Generate polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(input_lengths)

    #Model fitting

    print("Fitting Linear regression!")
    model  = LinearRegression()
    model.fit(X_poly, output_lengths)

    train_predictions = model.predict(X_poly)

    #train error
    mse = mean_squared_error(output_lengths, train_predictions)
    r_squared = r2_score(output_lengths, train_predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r_squared}")

    #val error
    val_input_lengths, val_output_lengths = get_data(val_data)
    val_input_lengths = np.array(val_input_lengths).reshape(-1, 1)
    val_output_lengths = np.array(val_output_lengths)

    val_poly = poly_features.transform(val_input_lengths)

    val_predictions = model.predict(val_poly)
    
    mse = mean_squared_error(val_output_lengths, val_predictions)
    r_squared = r2_score(val_output_lengths, val_predictions)
    
    print(f"Validation Mean Squared Error: {mse}")
    print(f"Validation R-squared: {r_squared}")

    for i in range(val_output_lengths.shape[0]):
        print(f"Input: {val_input_lengths[i][0]} Prediction:{val_predictions[i]} Actual:{val_output_lengths[i]}")


    filename = f'length_model_degree_{degree}'
    model_filename = f'{args.exp_dir}/{filename}'
    dump(model, model_filename)

    poly_filename = f'poly_feature_{degree}'
    poly_model_filename = f'{args.exp_dir}/{poly_filename}'
    dump(poly_features, poly_model_filename)

if __name__ == "__main__":
    main()