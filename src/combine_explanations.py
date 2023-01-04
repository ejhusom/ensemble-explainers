#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combine explanations from ensemble.

Author:
    Erik Johannes Husom

Created:
    2023-01-04 onsdag 12:46:14 

"""
import pandas as pd

from config import FEATURES_PATH


def combine_explanations(
        feature_importances,
        method="avg"
    ):
    """Combine explanations from ensemble.

    Args:
        feature_importances: DatFrame containing the feature importances for
            all models in ensemble. Rows: Models. Columns: Features.

    """


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


if __name__ == '__main__':

    feature_importances = pd.read_csv(FEATURES_PATH /
            "feature_importances.csv", index_col=0)

    # Piston rod:
    inadequate_models = ["dt", "gb", "sgd"]

    # Broaching:
    # inadequate_models = ["lgbm", "sgd"]

    # Delete rows of the models in inadequate_models
    for index, row in feature_importances.iterrows():
        if index.split("_")[-1] in inadequate_models:
            feature_importances.drop(index, inplace=True)

    print(feature_importances)
    
    combine_explanations(feature_importances)
