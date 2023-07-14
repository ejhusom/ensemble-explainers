#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-liner describing module.

Author:
    Erik Johannes Husom

Created:
    2021

"""
from config import *



def generate_explanation_report():

    with open(PLOTS_PATH / "prediction.html", "r") as infile:
        prediction_plot = infile.read()

    with open(PLOTS_PATH / "feature_importances.html", "r") as infile:
        feature_importances_plot = infile.read()

    html = "<html>"

    # html += f"<embed type='text/html' src='{'prediction.html'}' width='101%' height='500'>"
    html += prediction_plot

    html += "</html>"

    with open("assets/plots/report.html", "w") as outfile:
        outfile.write(html)

if __name__ == '__main__':
    generate_explanation_report()
