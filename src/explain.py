#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explain predictions of machine learning model.

Author:
    Erik Johannes Husom

Created:
    2022-11-28 mandag 16:01:00 

Notes:

    - https://github.com/slundberg/shap/issues/213

"""
import csv
import json
import os
import sys

import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import yaml
from joblib import load
from lime import submodular_pick
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras import models

tf.compat.v1.disable_v2_behavior()

from config import (
    ADEQUATE_MODELS_PATH,
    DL_METHODS,
    FEATURES_PATH,
    INPUT_FEATURES_PATH,
    INPUT_FEATURES_SEQUENCE_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    MODELS_PATH,
    NON_DL_METHODS,
    NON_SEQUENCE_LEARNING_METHODS,
    PLOTS_PATH,
    SEQUENCE_LEARNING_METHODS,
)

colors = []


for l in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, l))

for l in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, l))

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
    explanation_method = params["explanation_method"]
    learning_method = params_train["learning_method"]
    ensemble = params_train["ensemble"]
    seed = params["seed"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]
    
    if seed is None:
        seed = np.random.randint(0)

    # Load training data
    train = np.load(train_filepath)
    X_train = train["X"]

    test = np.load(test_filepath)
    X_test = test["X"]
    y_test = test["y"]

    # Read name of input columns
    input_columns = pd.read_csv(INPUT_FEATURES_PATH, header=None)

    # Convert the input columns into a list
    input_columns = input_columns.iloc[1:, 1].to_list()

    if ensemble:
        with open(ADEQUATE_MODELS_PATH / "adequate_models.json", "r") as f:
            adequate_models = json.load(f)

        model_names = []
        adequate_methods = []

        for f in os.listdir(MODELS_PATH):
            if f.startswith("model"):
                model_names.append(f)

        model_names = sorted(model_names)

        feature_importances = []

        # Variables for saving n most important features for all models
        n = 10
        # importance_table_rows = []
        sorted_feature_importances = {}

        for name in model_names:
            if name in adequate_models.keys():
                method = os.path.splitext(name)[0].split("_")[-1]
                adequate_methods.append(method)
                if method in DL_METHODS:
                    model = models.load_model(MODELS_PATH / name)
                else:
                    model = load(MODELS_PATH / name)

                print(f"Explaining {method}")
                shap_values = explain_predictions(
                    model,
                    X_train,
                    X_test,
                    window_size,
                    method,
                    input_columns,
                    number_of_background_samples,
                    number_of_summary_samples,
                    make_plots=False,
                    seed=2020,
                    classification=classification,
                    explanation_method=explanation_method,
                )

                feature_importance = get_feature_importance(shap_values,
                        input_columns, label=method)

                # Scale feature importances to range of [0, 1]
                feature_importance = feature_importance.div(feature_importance.sum(axis=0), axis=1)

                # Save the ten most important features
                sorted_feature_importance = feature_importance.sort_values(
                        by=f"feature_importance_{method}", 
                        ascending=False
                )
                sorted_feature_importance.to_csv(FEATURES_PATH /
                        f"sorted_feature_importance_{method}.csv")

                sorted_feature_importances[method] = sorted_feature_importance


                feature_importance = feature_importance.transpose()
                feature_importances.append(feature_importance)

        print(feature_importances)
        feature_importances = pd.concat(feature_importances)

        feature_importances.to_csv(FEATURES_PATH / "feature_importances.csv")

        pd.options.plotting.backend = "plotly"
        fig = feature_importances.plot.bar()
        fig.write_html(str(PLOTS_PATH / "feature_importances.html"))
        fig.show()


        adequate_methods = sorted(adequate_methods)

        n_tables = 2
        n_methods_per_table = len(adequate_methods) // n_tables
        overflow_methods = len(adequate_methods) % n_tables
        
        for t in range(n_tables):
            first_method = t * n_methods_per_table

            if overflow_methods > 0 and t == n_tables - 1:
                last_method = (t+1) * n_methods_per_table + overflow_methods
            else:
                last_method = (t+1) * n_methods_per_table

            header_row = ""

            # Create header row for importance_table.
            for method in adequate_methods[first_method:last_method]:
                header_row += f" & {method} " + r"Feature & $\bar{S}$ "

            rows = [header_row]

            for i in range(n):
                row = f"{i+1} "
                for method in adequate_methods[first_method:last_method]:
                    df = sorted_feature_importances[method]

                    feature = df.index[i].replace("_", r"\_")
                    value = df[f"feature_importance_{method}"][i]
                    row += r" & \texttt{" + feature + "}" + f" & {round(value, 3)} "

                rows.append(row)

            
            importance_table = "\\\\ \n".join(rows) + "\\\\"

            with open(FEATURES_PATH / f"importance_table_{t}.tex", "w") as f:
                f.write(importance_table)

        generate_explanation_report()

    else:

        if learning_method in NON_DL_METHODS:
            model = load(model_filepath)
        else:
            model = models.load_model(model_filepath)

        explain_predictions(
            model,
            X_train,
            X_test,
            window_size,
            learning_method,
            input_columns,
            number_of_background_samples,
            number_of_summary_samples,
            classification=classification,
            explanation_method=explanation_method,
        )

    # Delete rows of the models in inadequate_models
    for index, row in feature_importances.iterrows():
        if index.split("_")[-1] not in adequate_methods:
            feature_importances.drop(index, inplace=True)

    combine_explanations(feature_importances)

def get_feature_importance(shap_values, column_names, label=""):
    # Source: https://github.com/slundberg/shap/issues/632

    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(column_names, vals)),columns=['col_name',f"feature_importance_{label}"])
    feature_importance.sort_values(by=[f"feature_importance_{label}"], ascending=False, inplace=True)
    feature_importance = feature_importance.set_index("col_name")

    return feature_importance

def explain_predictions(
    model,
    X_train,
    X_test,
    window_size,
    learning_method,
    input_columns,
    number_of_background_samples,
    number_of_summary_samples,
    make_plots=True,
    seed=2022,
    explanation_method="shap",
    classification=False,
):

    if explanation_method == "shap":
        xai_values = explain_predictions_shap(
            model,
            X_train,
            X_test,
            window_size,
            learning_method,
            input_columns,
            number_of_background_samples,
            number_of_summary_samples,
            make_plots=False,
            seed=2020,
        )
    elif explanation_method == "lime":
        xai_values = explain_predictions_lime(
            model,
            X_train,
            X_test,
            window_size,
            learning_method,
            input_columns,
            number_of_background_samples,
            number_of_summary_samples,
            make_plots=False,
            seed=2020,
            classification=classification,
        )
    else:
        raise NotImplementedError(f"Explanation method {explanation_method} is not implemented.")

    return xai_values

def explain_predictions_shap(
    model,
    X_train,
    X_test,
    window_size,
    learning_method,
    input_columns,
    number_of_background_samples,
    number_of_summary_samples,
    make_plots=True,
    seed=2022,
):

    X_test_summary = shap.sample(X_test, number_of_summary_samples,
            random_state=seed)

    if learning_method in NON_SEQUENCE_LEARNING_METHODS:
        if window_size > 1:
            input_columns_sequence = []

            for c in input_columns:
                for i in range(window_size):
                    input_columns_sequence.append(c + f"_{i}")

            input_columns = input_columns_sequence

        # Extract a summary of the training inputs, to reduce the amount of
        # compute needed to use SHAP
        k = np.min([X_train.shape[0], 50])
        X_train_background = shap.kmeans(X_train, k)

        # Use a SHAP explainer on the summary of training inputs
        explainer = shap.KernelExplainer(model.predict, X_train_background)
        print(learning_method)
        # explainer = shap.TreeExplainer(model, X_train_background)

        print("1 ============")
        # Single prediction explanation
        single_sample = X_test[0]
        print(single_sample.shape)
        single_shap_value = explainer.shap_values(single_sample)
        shap_values = explainer.shap_values(X_test_summary)

        if type(single_shap_value) == list:
            single_shap_value = single_shap_value[0]
            shap_values = shap_values[0]

        if make_plots:
            # SHAP force plot: Single prediction
            shap_force_plot_single = shap.force_plot(
                explainer.expected_value,
                single_shap_value,
                np.around(single_sample),
                show=True,
                feature_names=input_columns,
            )
            shap.save_html(
                str(PLOTS_PATH) + "/shap_force_plot_single.html", shap_force_plot_single
            )

            # SHAP force plot: Multiple prediction
            shap_force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values,
                X_test_summary,
                show=True,
                feature_names=input_columns,
            )
            shap.save_html(str(PLOTS_PATH) + "/shap_force_plot.html", shap_force_plot)

            # SHAP summary plot
            shap.summary_plot(
                shap_values,
                X_test_summary,
                feature_names=input_columns,
                plot_size=(8, 5),
                show=False,
                max_display=10,
            )
            plt.xticks(rotation = 45)
            plt.tight_layout()

            plt.savefig(
                PLOTS_PATH / "shap_summary_plot.png", bbox_inches="tight", dpi=300
            )
            # plt.show()
    else:
        # Extract a summary of the training inputs, to reduce the amount of
        # compute needed to use SHAP
        X_train_background = shap.sample(X_train, number_of_background_samples,
                random_state=seed)

        # Use a SHAP explainer on the summary of training inputs
        explainer = shap.DeepExplainer(model, X_train_background)

        # Single prediction explanation
        single_sample = X_test[:1]
        single_shap_value = explainer.shap_values(single_sample)[0]
        shap_values = explainer.shap_values(X_test_summary)[0]

        if make_plots:
            # SHAP force plot: Single prediction
            shap_force_plot_single = shap.force_plot(
                explainer.expected_value,
                shap_values[0, :],
                X_test_summary[0, :],
                feature_names=input_columns,
            )
            shap.save_html(
                str(PLOTS_PATH) + "/shap_force_plot_single.html", shap_force_plot_single
            )

            # Expand dimensions with 1 in order to plot. The built-in
            # image_plot of the shap library requires channel as one of the
            # dimensions in the input arrays. Therefore we add one dimension to
            # the arrays to make it seem like it is an image with one array.
            X_test_summary = np.expand_dims(X_test_summary, axis=3)
            shap_values = np.expand_dims(shap_values, axis=3)

            # SHAP image plot
            number_of_input_sequences = 5
            shap_image_plot = shap.image_plot(
                shap_values[:number_of_input_sequences, :],
                X_test_summary[:number_of_input_sequences, :],
                show=False,
            )
            plt.savefig(
                PLOTS_PATH / "shap_image_plot.png", bbox_inches="tight", dpi=300
            )

    return shap_values

def explain_predictions_lime(
    model,
    X_train,
    X_test,
    window_size,
    learning_method,
    input_columns,
    number_of_background_samples,
    number_of_summary_samples,
    make_plots=True,
    seed=2022,
    classification=False,
):

    if classification:
        mode = "classification"
    else:
        mode = "regression"

    if window_size == 1:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=input_columns,
                mode=mode,
                discretize_continuous=False,
        )
    else:
        lime_explainer = lime.lime_tabular.LimeRecurrentTabularExplainer(
                X_train,
                feature_names=input_columns,
                mode=mode,
                discretize_continuous=False,
        )

    sp_obj = lime.submodular_pick.SubmodularPick(lime_explainer, X_train,
        model.predict, sample_size=number_of_background_samples)

    #Making a dataframe of all the explanations of sampled points
    xai_values = pd.DataFrame([dict(this.as_list()) for this in sp_obj.explanations]).fillna(0)

    if make_plots:
        ##Plotting the aggregate importances
        avg_xai_values = np.abs(xai_values).mean(axis=0).sort_values(ascending=False).head(
            25
        ).sort_values(ascending=True).plot(kind="barh")
        # plt.show()

        plt.savefig(
            PLOTS_PATH / "lime_summary_plot.png", bbox_inches="tight", dpi=300
        )

    return xai_values

def combine_explanations(
        feature_importances,
        method="avg"
    ):
    """Combine explanations from ensemble.

    Args:
        feature_importances: DatFrame containing the feature importances for
            all models in ensemble. Rows: Models. Columns: Features.

    """

    feature_importances.fillna(0, inplace=True) 

    if method == "avg":
        combined_feature_importances = feature_importances.mean(0,
                numeric_only=True)
    else:
        print(" -- Using default combination method: average")
        combined_feature_importances = feature_importances.mean(0,
                numeric_only=True)

    # Save the ten most important features
    sorted_combined_feature_importances = combined_feature_importances.sort_values(
            ascending=False
    )
    sorted_combined_feature_importances.to_csv(FEATURES_PATH /
            "sorted_combined_feature_importances.csv")

    print(sorted_combined_feature_importances)

    return sorted_combined_feature_importances

def generate_explanation_report():

    with open(PLOTS_PATH / "prediction.html", "r") as infile:
        prediction_plot = infile.read()

    with open(PLOTS_PATH / "feature_importances.html", "r") as infile:
        feature_importances_plot = infile.read()

    sorted_combined_feature_importances_filepath = FEATURES_PATH / "sorted_combined_feature_importances.csv"
    sorted_combined_feature_importances_table = generate_html_table(sorted_combined_feature_importances_filepath)

    html = "<html>\n"
    html += "<head>\n"
    html += "<meta charset='UTF-8'>"
    html += "<title>Model prediction and explanations</title>"
    html += "<link href='../../src/static/style.css' rel='stylesheet' type='text/css' title='Stylesheet'>"
    html += "<link rel='icon' type='image/png' href='../../src/static/favicon.png'>"
    html += "<script src='../../src/static/jquery.js'></script>"
    html += "</head>"

    html += "<body>"
    html += "<header>"
    html += "<div id=logoContainer>"
    html += "<img src='../../src/static/sintef-logo-centered-negative.svg' id=logo>"
    html += "<h1>Model prediction and explanations</h1>"
    html += "</div>"
    html += "<nav>"
    html += "    <a href='#prediction'>True vs predicted values</a>"
    html += "    <a href='#featureimportanceschart'>Feature importances chart</a>"
    html += "    <a href='#featureimportancestable'>Feature importances table</a>"
    html += "</nav>"
    html += "</header>"

    html += "<div class=box>"
    html += "<h2 id='featureimportancestable'>Feature importances table</h2>"
    html += "<div class=overviewTable>"
    html += sorted_combined_feature_importances_table
    html += "</div>"
    html += "</div>"

    html += "<div class=box>"
    html += "<h2 id='prediction'>True vs predicted values</h2>"
    html += prediction_plot
    html += "</div>"

    html += "<div class=box>"
    html += "<h2 id='featureimportanceschart'>Feature importances chart</h2>"
    html += feature_importances_plot
    html += "</div>"

    html += "</body>"
    html += "</html>"

    with open("assets/plots/report.html", "w") as outfile:
        outfile.write(html)

def generate_html_table(csv_file):
    table_html = "<table>\n"
    with open(csv_file, "r") as file:
        csv_reader = csv.reader(file)
        header_row = next(csv_reader)
        table_html += "  <thead>\n"
        table_html += "    <tr>\n"
        table_html += f"      <th>Feature name</th>\n"
        table_html += f"      <th>Impact score</th>\n"
        table_html += "    </tr>\n"
        table_html += "  </thead>\n"

        table_html += "  <tbody>\n"
        for row in csv_reader:
            table_html += "    <tr>\n"
            for i, cell in enumerate(row):
                if i == 1:  # Format numbers in the second column
                    formatted_cell = "{:.3f}".format(float(cell))
                else:
                    formatted_cell = cell
                table_html += f"      <td>{formatted_cell}</td>\n"
            table_html += "    </tr>\n"
        table_html += "  </tbody>\n"
    table_html += "</table>"
    return table_html

if __name__ == "__main__":

    if len(sys.argv) < 3:
        try:
            explain(
                "assets/models/",
                "assets/data/combined/train.npz",
                "assets/data/combined/test.npz",
            )
        except:
            print("Could not find model and test set.")
            sys.exit(1)
    else:
        explain(sys.argv[1], sys.argv[2], sys.argv[3])

