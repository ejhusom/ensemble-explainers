#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explain predictions of machine learning model.

Author:
    Erik Johannes Husom

Created:
    2022-11-28 mandag 16:01:00 

"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import yaml

from joblib import load
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras import models
tf.compat.v1.disable_v2_behavior()

from config import (
    ADEQUATE_MODELS_PATH,
    DATA_PATH,
    DL_METHODS,
    INPUT_FEATURES_PATH,
    INPUT_FEATURES_SEQUENCE_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    MODELS_PATH,
    NON_DL_METHODS,
    NON_SEQUENCE_LEARNING_METHODS,
    OUTPUT_FEATURES_PATH,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH,
    PREDICTIONS_PATH,
    SEQUENCE_LEARNING_METHODS
)

colors = []


for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))

for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))

red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

def explain(
        model_filepath,
        train_filepath,
        test_filepath,
    ):
    """Create explanations for predictions made by one or multiple models.

    model_filepath (str): Path to folder containing models.
    train_filepath (str): Path to training data set.
    test_filepath (str): Path to testing data set.

    """


    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["explain"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]
    window_size = yaml.safe_load(open("params.yaml"))["sequentialize"]["window_size"]
    number_of_background_samples = params["number_of_background_samples"]
    number_of_summary_samples = params["number_of_summary_samples"]
    learning_method = params_train["learning_method"]
    ensemble = params_train["ensemble"]

    # Load training data
    train = np.load(train_filepath)
    X_train = train["X"]

    test = np.load(test_filepath)
    X_test = test["X"]
    y_test = test["y"]

    # Read name of input columns
    input_columns = pd.read_csv(INPUT_FEATURES_PATH, header=None)

    # Convert the input columns into a list
    input_columns = input_columns.iloc[1:,1].to_list()

    if ensemble:
        with open(ADEQUATE_MODELS_PATH / "adequate_models.json", "r") as f:
            adequate_models = json.load(f)

        model_names = []

        for f in os.listdir(MODELS_PATH):
            if f.startswith("model"):
                model_names.append(f)
            
        model_names = sorted(model_names)

        for name in model_names:
            print(adequate_models.keys())
            if name in adequate_models.keys():
                method = os.path.splitext(name)[0].split("_")[-1]
                if method in DL_METHODS:
                    model = models.load_model(MODELS_PATH / name)
                else:
                    model = load(MODELS_PATH / name)

                explain_predictions(
                        model,
                        X_train,
                        X_test,
                        window_size,
                        learning_method,
                        input_columns,
                        number_of_background_samples,
                        number_of_summary_samples,
                )

        return 0

    if learning_method in NON_DL_METHODS:
        model = load(model_filepath)
        y_pred = model.predict(X_test)
    else:
        model = models.load_model(model_filepath)
        y_pred = model.predict(X_test)


    explain_predictions(
            model,
            X_train,
            X_test,
            window_size,
            learning_method,
            input_columns,
            number_of_background_samples,
            number_of_summary_samples,
    )


def explain_predictions(
        model,
        X_train,
        X_test,
        window_size,
        learning_method,
        input_columns,
        number_of_background_samples,
        number_of_summary_samples,
        make_plots=True
    ):

    if learning_method in NON_SEQUENCE_LEARNING_METHODS:
        if window_size > 1:
            input_columns_sequence = []

            for c in input_columns:
                for i in range(window_size):
                    input_columns_sequence.append(c + f"_{i}")

            input_columns = input_columns_sequence

        # Extract a summary of the training inputs, to reduce the amount of
        # compute needed to use SHAP
        X_train_background = shap.kmeans(X_train, number_of_background_samples)
        X_test_summary = shap.sample(X_test, number_of_summary_samples)

        # Use a SHAP explainer on the summary of training inputs
        explainer = shap.KernelExplainer(model.predict, X_train_background)

        # Single prediction explanation
        single_sample = X_test[0]
        single_shap_value = explainer.shap_values(single_sample)
        shap_values = explainer.shap_values(X_test_summary)

        if type(single_shap_value) == list:
            single_shap_value = single_shap_value[0]
            shap_values = shap_values[0]


        if make_plots:
            # SHAP force plot: Single prediction
            shap_force_plot_single = shap.force_plot(explainer.expected_value, single_shap_value,
                    np.around(single_sample), show=True, feature_names=input_columns)
            shap.save_html(str(PLOTS_PATH) + "/shap_force_plot_single.html",
                    shap_force_plot_single)

            # SHAP force plot: Multiple prediction
            shap_force_plot = shap.force_plot(explainer.expected_value, shap_values,
                    X_test_summary, show=True, feature_names=input_columns)
            shap.save_html(str(PLOTS_PATH) + "/shap_force_plot.html", shap_force_plot)

            # SHAP summary plot
            shap.summary_plot(shap_values, X_test_summary,
                    feature_names=input_columns, plot_size=(8,5), show=False)
            plt.savefig(PLOTS_PATH / "shap_summary_plot.png", bbox_inches='tight', dpi=300)
    else:
        # Extract a summary of the training inputs, to reduce the amount of
        # compute needed to use SHAP
        X_train_background = shap.sample(X_train, number_of_background_samples)
        X_test_summary = shap.sample(X_test, number_of_summary_samples)

        # Use a SHAP explainer on the summary of training inputs
        explainer = shap.DeepExplainer(model, X_train_background)

        # Single prediction explanation
        single_sample = X_test[:1]
        single_shap_value = explainer.shap_values(single_sample)[0]
        shap_values = explainer.shap_values(X_test_summary)[0]

        if make_plots:
            # SHAP force plot: Single prediction
            shap_force_plot_single = shap.force_plot(explainer.expected_value, shap_values[0,:],
                    X_test_summary[0,:], feature_names=input_columns)
            shap.save_html(str(PLOTS_PATH) + "/shap_force_plot_single.html",
                    shap_force_plot_single)

            X_test_summary = np.expand_dims(X_test_summary, axis=3)
            shap_values = np.expand_dims(shap_values, axis=3)
            # SHAP image plot
            # shap_image_plot = shap.image_plot(shap_values, X_test_summary,
            # shap_image_plot = image_plot(shap_values[0,:], X_test_summary[0,:],
            shap_image_plot = shap.image_plot(shap_values[:5,:], X_test_summary[:5,:],
                    show=False)
            plt.savefig(PLOTS_PATH / "shap_image_plot.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        try:
            explain(
                "assets/models/model.h5",
                "assets/data/combined/train.npz",
                "assets/data/combined/test.npz"
            )
        except:
            print("Could not find model and test set.")
            sys.exit(1)
    else:
        explain(sys.argv[1], sys.argv[2], sys.argv[3])

