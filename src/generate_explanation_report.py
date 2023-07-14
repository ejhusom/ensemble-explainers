#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-liner describing module.

Author:
    Erik Johannes Husom

Created:
    2021

"""
import csv

from config import *

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
    html += "<title>ERDRE - Virtual sensors</title>"
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


# def generate_html_table(csv_file):
#     table_html = "<table>\n"
#     with open(csv_file, "r") as file:
#         csv_reader = csv.reader(file)
#         for row in csv_reader:
#             table_html += "  <tr>\n"
#             for cell in row:
#                 table_html += f"    <td>{cell}</td>\n"
#             table_html += "  </tr>\n"
#     table_html += "</table>"
#     return table_html

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


if __name__ == '__main__':
    generate_explanation_report()
