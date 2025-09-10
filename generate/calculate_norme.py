"""
Calculate the norm for different groups (agé/jeune)

data_agés_E / data_jeunes_E : without induction

for the moment: function only for data_agés_E !
"""

import yaml
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

path_to_yaml = "../settings.yaml"

with open(path_to_yaml, 'r') as file:
    config = yaml.safe_load(file)

# Construct full path to the CSV
stats_dir = config["paths"]["statistics"]
csv_filename = config["file_patterns"]["user_stats"]
UserInfoStatPath = os.path.join(stats_dir, csv_filename)  # Combines paths safely

df = pd.read_csv(UserInfoStatPath, delimiter=",") # header=None

# add a column for sum_duration_speak_per_minute (sec/min) = (min/hour)
df["sum_duration_speak_per_minute"] = (df["sum_duration_speak"] / df["duration_session"])

# function to categorize group with / without induction
def tran_cat_to_num(df):
    if df['induction_group'] == 'E':
        return "data_agés_E"
    elif df['induction_group'] != 'E':
        return "data_agés_ABCD"

df['nom_data_fichier_new']=df.apply(tran_cat_to_num, axis=1)

df_data_agés_E = (df[df["nom_data_fichier_new"] == "data_agés_E"])
# len(df_data_agés_E) # 39
# df_data_agés_ABCD = (df[df["nom_data_fichier_new"] =="data_agés_ABCD"])
# len(df_data_agés_ABCD) # 31

def get_mean_std_each_variable(var):
    """
    Around 68% of values are within 1 standard deviation from the mean.
    Around 95% of values are within 2 standard deviations from the mean.
    Around 99.7% of values are within 3 standard deviations from the mean.
    """
    mean = df_data_agés_E["{}".format(var)].mean()
    std = df_data_agés_E["{}".format(var)].std()
    left = round(mean - std*2, 2)
    right = round(mean + std*2, 2)
    norm = "{} - {}".format(left, right) # 95%
    return norm

def get_quartile_range(variable_name, decimal_places=0):
    """
    Calculate the interquartile range (IQR) for a specified variable.

    Args:
        variable_name (str): Name of the column/variable to analyze
        decimal_places (int): Number of decimal places to round to (0 for integer output)

    Returns:
        str: Formatted string showing the quartile range "Q1-Q3" without decimals when decimal_places=0
    """
    series = df_data_agés_E[variable_name]
    quantile_25 = series.quantile(0.25)
    quantile_75 = series.quantile(0.75)

    if decimal_places == 0:
        left = int(round(quantile_25))
        right = int(round(quantile_75))
        return f"{left} - {right}"
    else:
        left = round(quantile_25, decimal_places)
        right = round(quantile_75, decimal_places)
        return f"{left} - {right}"

def compare_value_to_norm(value, norm):
    import colorama
    from colorama import Fore
    # comparison = Fore.RED + "🠕"
    norm_left = float(norm.split(" - ")[0])
    norm_right = float(norm.split(" - ")[1])
    if norm_left <= value <= norm_right:
        comparison = ""
    elif value < norm_left: 
        comparison = "&#129047;" # Leftwards White Arrow ⇦
    elif value > norm_right:
        comparison = "&#129045;" # Rightwards White Arrow ⇨ 🠕
    return comparison
