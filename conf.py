# %%
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import confusion_matrix


# %%
# Define Parse Function to streamline output
def parse(x):
    if x in [0, "0", "False", "false", False, "Nein", "nein"]:
        return "False"
    elif x in [1, "1", "True", "true", True, "Ja", "ja"]:
        return "True"
    else:
        return "False"

#%% 
# Define result json to df function to -ALTERNATIVE WITH CoT output-
# def result_json_to_df(json_path):
#     symptoms = [
#         "compl",
#     ]
#     with open(json_path, "r") as json_file:
#         records = []
#         for line in json_file:
#             try:
#                 llama_response = json.loads(line)
#                 compl = llama_response["content"]  # Extract the value from "content"
#                 report = llama_response.get("report", "")  # Extract the report, default to empty string if not present
#                 records.append((report, compl))
#             except json.JSONDecodeError:
#                 continue

#     pred_df = pd.DataFrame(records, columns=["report", "compl"])
#     #pred_df[compl] = pred_df[compl].applymap(parse)
#     return pred_df

# %%
# load ground truth data
#gt_df = pd.read_excel("/mnt/bulk/isabella/llamaproj/endo/reports_GT.xlsx")
gt_df = pd.read_csv("merged_GT_3.csv")
gt_df.head()

#%%
# load predicted df
filename="suizidal_llama2_sauerkraut_70b_cot_3s.csv"
pred_df = pd.read_csv(f"results\\{filename}")
pred_df.head()

#%%
# merge pred_df and gt_df on "report" column
df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])
df.head()

#%%
# Parse the columns compl and compl pred
df['suizidal'] = df['suizidal'].map(parse)
df['suizidal pred'] = df['suizidal pred'].map(parse)
df.head()

###########################################################################
# Confusion matrix

# %%
symptoms = ["suizidal"]
for symptom in symptoms:
    y_true = df[symptom]
    y_pred = df[f"{symptom} pred"]

    # Compute the confusion matrix (non-normalized for absolute numbers)
    cm_absolute = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix for fractions
    cm_normalized = cm_absolute.astype('float') / cm_absolute.sum(axis=1)[:, np.newaxis]

    # Convert to DataFrame for easier plotting
    cm_df = pd.DataFrame(cm_normalized, index=["False", "True"], columns=["False", "True"])

    # Create annotations combining absolute numbers and fractions
    annotations = [["{0:d}\n({1:.2f})".format(abs_num, frac) for abs_num, frac in zip(row_abs, row_frac)] 
                    for row_abs, row_frac in zip(cm_absolute, cm_normalized)]

    # Plotting the confusion matrix using Seaborn with increased font sizes
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm_df, annot=annotations, fmt="", cmap='Blues', vmin=0, vmax=1, annot_kws={"size": 28})

    plt.title(f'{filename}{symptom.capitalize()}', fontsize=28)
    plt.ylabel('Actual Values', fontsize=18)
    plt.xlabel('Predicted Values', fontsize=18)

    # Set the font size for the tick labels (both axes)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 28)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 28)

    # Increase font size of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)

    plt.show()
# %%
