#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explain predictions of machine learning model.

Author:
    Erik Johannes Husom

Created:
    2022-11-28 mandag 16:01:00 

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import yaml

from joblib import load
from tensorflow.keras import models
tf.compat.v1.disable_v2_behavior()

from config import (
    DATA_PATH,
    INPUT_FEATURES_PATH,
    INPUT_FEATURES_SEQUENCE_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    NON_DL_METHODS,
    NON_SEQUENCE_LEARNING_METHODS,
    OUTPUT_FEATURES_PATH,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH,
    PREDICTIONS_PATH,
    SEQUENCE_LEARNING_METHODS
)

def explain(
        model_filepath,
        train_filepath,
        test_filepath,
    ):

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["explain"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]
    window_size = yaml.safe_load(open("params.yaml"))["sequentialize"]["window_size"]
    onehot_encode_target = yaml.safe_load(open("params.yaml"))["clean"][
        "onehot_encode_target"
    ]
    number_of_background_samples = params["number_of_background_samples"]
    number_of_summary_samples = params["number_of_summary_samples"]
    learning_method = params_train["learning_method"]

    # Load training data
    train = np.load(train_filepath)
    X_train = train["X"]

    test = np.load(test_filepath)
    X_test = test["X"]
    y_test = test["y"]

    if learning_method in NON_DL_METHODS:
        model = load(model_filepath)
        y_pred = model.predict(X_test)
    else:
        model = models.load_model(model_filepath)
        y_pred = model.predict(X_test)

    # Read name of input columns
    input_columns = pd.read_csv(INPUT_FEATURES_PATH, header=None)

    # Convert the input columns into a list
    input_columns = input_columns.iloc[1:,1].to_list()

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
        X_test_summary = shap.sample(X_train, number_of_summary_samples)

        # Use a SHAP explainer on the summary of training inputs
        explainer = shap.KernelExplainer(model.predict, X_train_background)

        # Single prediction explanation
        single_sample = X_test[0]
        single_shap_value = explainer.shap_values(single_sample)
        shap_values = explainer.shap_values(X_test_summary)

        if type(single_shap_value) == list:
            single_shap_value = single_shap_value[0]
            shap_values = shap_values[0]

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
                feature_names=input_columns, plot_size=(8,5), show=True)
        plt.savefig(PLOTS_PATH / "shap_summary_plot.png", bbox_inches='tight', dpi=300)
    else:
        # Extract a summary of the training inputs, to reduce the amount of
        # compute needed to use SHAP
        X_train_background = shap.sample(X_train, number_of_background_samples)
        X_test_summary = shap.sample(X_train, number_of_summary_samples)

        # Use a SHAP explainer on the summary of training inputs
        # explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), X_train_background)
        explainer = shap.DeepExplainer(model, X_train_background)

        # Single prediction explanation
        single_sample = X_test[:1]
        single_shap_value = explainer.shap_values(single_sample)
        shap_values = explainer.shap_values(X_test_summary)

        print("===================")
        print(len(shap_values))

        single_shap_value = single_shap_value[0]
        shap_values = shap_values[0]


        print("--------------------")
        print(single_shap_value.shape)
        print(single_sample.shape)
        print(shap_values.shape)
        print(explainer.expected_value.shape)
        print(X_test_summary.shape)
        print(explainer.expected_value)
        print(explainer.expected_value[0])
        print("===================")

        # SHAP force plot: Single prediction
        shap_force_plot_single = shap.force_plot(explainer.expected_value, shap_values[0,:],
                X_test_summary[0,:], feature_names=input_columns)
        # shap.force_plot(explainer.expected_value[0], single_shap_value, single_sample)
                # np.around(single_sample), show=True, matplotlib=True,
                # feature_names=input_columns)
        shap.save_html(str(PLOTS_PATH) + "/shap_force_plot_single.html",
                shap_force_plot_single)

        # SHAP force plot: Multiple prediction
        # shap_force_plot = shap.force_plot(explainer.expected_value[0], shap_values,
        #         X_test_summary, show=True, feature_names=input_columns)
        # shap_force_plot = shap.force_plot(explainer.expected_value, shap_values, X_test_summary)
        # shap.save_html(str(PLOTS_PATH) + "/shap_force_plot.html", shap_force_plot)

        # SHAP summary plot
        # shap.summary_plot(shap_values, X_test_summary,
        #         feature_names=input_columns, plot_size=(8,5), show=False)
        # plt.savefig(PLOTS_PATH / "shap_summary_plot.png", bbox_inches='tight', dpi=300)

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
