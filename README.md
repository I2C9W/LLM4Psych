# LLM4Psych
# Detection of Suicidality Through Privacy-Preserving  Large Language Models
**Note: This documentation is currently under construction. Some sections may be updated or changed as development progresses.**

## General Setup Instructions

Before running the scripts, please ensure the following setup steps are completed:

1. **Python Installation**: Make sure Python is installed on your system. The scripts are compatible with Python 3.12.
2. **Dependency Installation**: Install the required Python packages. You can do this easily by using the `requirements.txt` file provided:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Place your dataset files in accessible paths on your system.

## Script-Specific Instructions

### Suicidality Extraction Script (`extractsuicidality.py`)
This Python script extracts and analyzes specific medical features from patient reports using a predefined prompt.
It runs on the basis of llama.cpp. Please set it up according to the respective github repository and start the llama server. (--> https://github.com/ggerganov/llama.cpp/)

#### Usage
Run the script from the command line by specifying the path to your input file data

### Confusion Matrix Analysis Script (`analysis.py`)

This Python script generates confusion matrices for machine learning model predictions, comparing predictions against a ground truth dataset to visualize the performance of a classification model.
Enter the path to your groundtruth file as well as the path to your model output file. 

#### Usage

Run the script from the command line by specifying the path to your ground truth data and predictions:

```bash
python confusionmatrix.py path/to/ground_truth.csv path/to/predictions.jsonl
```

### Accuracy Comparison Script (`accuracy_comparison.py`)
This Python script compares the accuracy of different machine learning models, calculating and visualizing the accuracy of each model for various symptoms.

#### Usage
Run the script from the command line with the path to your ground truth data:
    
```bash
python accuracy_comparison.py path/to/ground_truth.csv
 ```
