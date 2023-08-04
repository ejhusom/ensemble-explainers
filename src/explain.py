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
from tensorflow.keras import models

tf.compat.v1.disable_v2_behavior()

from config import (
    ADEQUATE_MODELS_FILE_PATH,
    DL_METHODS,
    EXPLANATION_METHODS,
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
from utils import Struct


class Explain:
    def __init__(
        self,
        model_filepath,
        train_filepath,
        test_filepath,
    ):
        self.model_filepath = model_filepath
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath

        # Read parameter file and convert to object
        self.params = Struct(yaml.safe_load(open("params.yaml")))

        if self.params.explain.seed is None:
            self.params.explain.seed = np.random.randint(0)

        # Load data
        self.train_data = np.load(self.train_filepath)
        self.X_train = self.train_data["X"]
        self.test_data = np.load(self.test_filepath)
        self.X_test = self.test_data["X"]
        self.y_test = self.test_data["y"]

        # Read name of input columns and convert to list
        self.input_columns = pd.read_csv(INPUT_FEATURES_PATH, index_col=0)
        self.input_columns = self.input_columns.values.flatten().tolist()

        if self.params.train.ensemble:
            self.explain_ensemble()
        else:
            feature_importances = []

            if self.params.train.learning_method in NON_DL_METHODS:
                model = load(self.model_filepath)
            else:
                model = models.load_model(self.model_filepath)

            if self.params.explain.explanation_method == "all":
                self.params.explain.explanation_method = EXPLANATION_METHODS

            if not isinstance(self.params.explain.explanation_method, list):
                self.params.explain.explanation_method = [
                    self.params.explain.explanation_method
                ]

            for explanation_method in self.params.explain.explanation_method:
                xai_values = self.explain_predictions(
                    model, explanation_method, make_plots=True
                )
                column_label = (
                    explanation_method + "_" + self.params.train.learning_method
                )
                feature_importance = get_feature_importance(
                    xai_values, label=column_label
                )

                # Scale feature importances to range of [0, 1]
                feature_importance = feature_importance.div(
                    feature_importance.sum(axis=0), axis=1
                )

                sorted_feature_importance = feature_importance.sort_values(
                    by=f"feature_importance_{column_label}", ascending=False
                )
                sorted_feature_importance.to_csv(
                    FEATURES_PATH / f"sorted_feature_importance_{column_label}.csv"
                )

                feature_importance = feature_importance.transpose()
                feature_importances.append(feature_importance)

            # Concat feature importance dataframe for all learning methods
            feature_importances = pd.concat(feature_importances)
            feature_importances.to_csv(FEATURES_PATH / "feature_importances.csv")

            pd.options.plotting.backend = "plotly"
            fig = feature_importances.plot.bar()
            fig.write_html(str(PLOTS_PATH / "feature_importances.html"))
            fig.show()

            generate_explanation_report()

    def explain_ensemble(self):
        with open(ADEQUATE_MODELS_FILE_PATH, "r") as f:
            self.adequate_models = json.load(f)

        model_names = []
        adequate_methods = []

        for f in os.listdir(MODELS_PATH):
            if f.startswith("model"):
                model_names.append(f)

        model_names = sorted(model_names)

        feature_importances = []

        for name in model_names:
            if name in self.adequate_models.keys():
                method = os.path.splitext(name)[0].split("_")[-1]
                adequate_methods.append(method)

                # Load model (different methods depending on model type)
                if method in DL_METHODS:
                    model = models.load_model(MODELS_PATH / name)
                else:
                    model = load(MODELS_PATH / name)

                print(f"Explaining {method}")

                # If user has chosen to use all explanation methods, set the
                # parameter accordingly
                if self.params.explain.explanation_method == "all":
                    self.params.explain.explanation_method = EXPLANATION_METHODS

                if not isinstance(self.params.explain.explanation_method, list):
                    self.params.explain.explanation_method = [
                        self.params.explain.explanation_method
                    ]
                for explanation_method in self.params.explain.explanation_method:
                    xai_values = self.explain_predictions(
                        model, explanation_method, make_plots=True
                    )
                    column_label = explanation_method + "_" + method
                    feature_importance = get_feature_importance(
                        xai_values, label=column_label
                    )

                    d = get_directional_feature_importance(xai_values,
                            label=column_label)

                    feature_importance.to_csv(f"t1_{column_label}.csv")
                    d.to_csv(f"t2_{column_label}.csv")

                    # Scale feature importances to range of [0, 1]
                    feature_importance = feature_importance.div(
                        feature_importance.sum(axis=0), axis=1
                    )

                    sorted_feature_importance = feature_importance.sort_values(
                        by=f"feature_importance_{column_label}", ascending=False
                    )
                    sorted_feature_importance.to_csv(
                        FEATURES_PATH / f"sorted_feature_importance_{column_label}.csv"
                    )

                    feature_importance = feature_importance.transpose()
                    feature_importances.append(feature_importance)

        # Concat feature importance dataframe for all learning methods
        feature_importances = pd.concat(feature_importances)
        feature_importances.to_csv(FEATURES_PATH / "feature_importances.csv")

        pd.options.plotting.backend = "plotly"
        fig = feature_importances.plot.bar(
                labels=dict(index="", value="Feature importance (%)")
        )
        fig.write_html(str(PLOTS_PATH / "feature_importances.html"))
        fig.show()

        adequate_methods = sorted(adequate_methods)

        # generate_ensemble_explanation_tables(sorted_combined_feature_importances,
        #         adequate_methods)

        generate_explanation_report()

        # Delete rows of the models in inadequate_models
        for index, row in feature_importances.iterrows():
            if index.split("_")[-1] not in adequate_methods:
                feature_importances.drop(index, inplace=True)

        combine_ensemble_explanations(feature_importances)

    def explain_predictions(
        self,
        model,
        method="shap",
        make_plots=True,
    ):
        if method == "shap":
            xai_values = self.explain_predictions_shap(
                model,
                make_plots=make_plots,
            )
        elif method == "lime":
            xai_values = self.explain_predictions_lime(
                model,
                make_plots=make_plots,
            )
        else:
            raise NotImplementedError(
                f"Explanation method {method} is not implemented."
            )

        return xai_values

    def explain_predictions_shap(self, model, make_plots=False):
        X_test_summary = shap.sample(
            self.X_test,
            self.params.explain.number_of_summary_samples,
            random_state=self.params.explain.seed,
        )

        if self.params.train.learning_method in NON_SEQUENCE_LEARNING_METHODS:
            if self.params.sequentialize.window_size > 1:
                input_columns_sequence = []

                for c in self.input_columns:
                    for i in range(self.params.sequentialize.window_size):
                        input_columns_sequence.append(c + f"_{i}")

                self.input_columns = input_columns_sequence

            # Extract a summary of the training inputs, to reduce the amount of
            # compute needed to use SHAP
            k = np.min([self.X_train.shape[0], 50])
            X_train_background = shap.kmeans(self.X_train, k)

            # Use a SHAP explainer on the summary of training inputs
            explainer = shap.KernelExplainer(model.predict, X_train_background)
            print(self.params.train.learning_method)
            # explainer = shap.TreeExplainer(model, X_train_background)

            # Single prediction explanation
            single_sample = self.X_test[0]
            single_shap_value = explainer.shap_values(single_sample)
            # Shap values for summary of test data
            shap_values = explainer.shap_values(X_test_summary)

            if isinstance(single_shap_value, list):
                single_shap_value = single_shap_value[0]
                shap_values = shap_values[0]

            if make_plots:
                # SHAP force plot: Single prediction
                shap_force_plot_single = shap.force_plot(
                    explainer.expected_value,
                    single_shap_value,
                    np.around(single_sample),
                    show=True,
                    feature_names=self.input_columns,
                )
                shap.save_html(
                    str(PLOTS_PATH) + "/shap_force_plot_single.html",
                    shap_force_plot_single,
                )

                # SHAP force plot: Multiple prediction
                shap_force_plot = shap.force_plot(
                    explainer.expected_value,
                    shap_values,
                    X_test_summary,
                    show=True,
                    feature_names=self.input_columns,
                )
                shap.save_html(
                    str(PLOTS_PATH) + "/shap_force_plot.html", shap_force_plot
                )

                # SHAP summary plot
                shap.summary_plot(
                    shap_values,
                    X_test_summary,
                    feature_names=self.input_columns,
                    plot_size=(8, 5),
                    show=False,
                    max_display=10,
                )
                plt.xticks(rotation=45)
                plt.tight_layout()

                plt.savefig(
                    PLOTS_PATH / "shap_summary_plot.png", bbox_inches="tight", dpi=300
                )

                # for feature in self.input_columns:
                #     shap.dependence_plot(
                #         feature,
                #         shap_values,
                #         X_test_summary,
                #         feature_names=self.input_columns,
                #         # plot_size=(8, 5),
                #         show=False,
                #     )
                #     plt.savefig(
                #         PLOTS_PATH / f"shap_dependence_plot_{feature}.png", bbox_inches="tight", dpi=300
                #     )
                #     plt.show()
        else:
            # Extract a summary of the training inputs, to reduce the amount of
            # compute needed to use SHAP
            X_train_background = shap.sample(
                self.X_train,
                self.params.explain.number_of_background_samples,
                random_state=self.params.explain.seed,
            )

            # Use a SHAP explainer on the summary of training inputs
            explainer = shap.DeepExplainer(model, X_train_background)

            # Single prediction explanation
            single_sample = self.X_test[:1]
            single_shap_value = explainer.shap_values(single_sample)[0]
            shap_values = explainer.shap_values(X_test_summary)[0]

            if make_plots:
                # SHAP force plot: Single prediction
                shap_force_plot_single = shap.force_plot(
                    explainer.expected_value,
                    shap_values[0, :],
                    X_test_summary[0, :],
                    feature_names=self.input_columns,
                )
                shap.save_html(
                    str(PLOTS_PATH) + "/shap_force_plot_single.html",
                    shap_force_plot_single,
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

        shap_values = pd.DataFrame(shap_values, columns=self.input_columns).sort_index(
            axis=1
        )

        return shap_values

    def explain_predictions_lime(self, model, make_plots=False):
        if self.params.clean.classification:
            mode = "classification"
        else:
            mode = "regression"

        if self.params.sequentialize.window_size == 1:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_test,
                feature_names=self.input_columns,
                mode=mode,
                discretize_continuous=False,
            )
        else:
            lime_explainer = lime.lime_tabular.LimeRecurrentTabularExplainer(
                self.X_test,
                feature_names=self.input_columns,
                mode=mode,
                discretize_continuous=False,
            )

        sp_obj = lime.submodular_pick.SubmodularPick(
            lime_explainer,
            self.X_test,
            model.predict,
            sample_size=self.params.explain.number_of_background_samples,
            num_features=self.X_test.shape[-1],
        )

        # Making a dataframe of all the explanations of sampled points.
        xai_values = (
            pd.DataFrame([dict(this.as_list()) for this in sp_obj.explanations])
            .fillna(0)
            .sort_index(axis=1)
        )

        if make_plots:
            # Plotting the aggregate importances
            avg_xai_values = (
                np.abs(xai_values)
                .mean(axis=0)
                .sort_values(ascending=False)
                .head(25)
                .sort_values(ascending=True)
                .plot(kind="barh")
            )
            # plt.show()

            plt.savefig(
                PLOTS_PATH / "lime_summary_plot.png", bbox_inches="tight", dpi=300
            )

        return xai_values


def get_feature_importance(xai_values, label=""):
    # Modified from: https://github.com/slundberg/shap/issues/632

    vals = np.abs(xai_values).mean(0)
    feature_importance = pd.DataFrame(
        list(zip(xai_values.columns.tolist(), vals)),
        columns=["col_name", f"feature_importance_{label}"],
    )

    # feature_importance.sort_values(
    #     by=[f"feature_importance_{label}"], ascending=False, inplace=True
    # )

    feature_importance = feature_importance.set_index("col_name")

    return feature_importance

def get_directional_feature_importance(xai_values, label=""):
    # Modified from: https://github.com/slundberg/shap/issues/632

    # vals = np.abs(xai_values).mean(0)
    vals = xai_values.mean(0)
    feature_importance = pd.DataFrame(
        list(zip(xai_values.columns.tolist(), vals)),
        columns=["col_name", f"feature_importance_{label}"],
    )

    # feature_importance.sort_values(
    #     by=[f"feature_importance_{label}"], ascending=False, inplace=True
    # )

    feature_importance = feature_importance.set_index("col_name")

    return feature_importance

def combine_ensemble_explanations(feature_importances, method="avg"):
    """Combine explanations from ensemble.

    Args:
        feature_importances: DatFrame containing the feature importances for
            all models in ensemble. Rows: Models. Columns: Features.

    """

    feature_importances.fillna(0, inplace=True)

    if method == "avg":
        combined_feature_importances = feature_importances.mean(0, numeric_only=True)
    else:
        print("Using default combination method: average")
        combined_feature_importances = feature_importances.mean(0, numeric_only=True)

    sorted_combined_feature_importances = combined_feature_importances.sort_values(
        ascending=False
    )
    sorted_combined_feature_importances.to_csv(
        FEATURES_PATH / "sorted_combined_feature_importances.csv"
    )

    return sorted_combined_feature_importances


def generate_explanation_report():
    with open(PLOTS_PATH / "prediction.html", "r") as infile:
        prediction_plot = infile.read()

    with open(PLOTS_PATH / "feature_importances.html", "r") as infile:
        feature_importances_plot = infile.read()

    sorted_combined_feature_importances_filepath = (
        FEATURES_PATH / "sorted_combined_feature_importances.csv"
    )
    sorted_combined_feature_importances_table = generate_html_table(
        sorted_combined_feature_importances_filepath
    )

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
            Explain(
                "assets/models/",
                "assets/data/combined/train.npz",
                "assets/data/combined/test.npz",
            )
        except:
            print("Could not find model and test set.")
            sys.exit(1)
    else:
        Explain(sys.argv[1], sys.argv[2], sys.argv[3])
