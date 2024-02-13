#%%
import pandas as pd
import json
import re
import chardet

#%%
#ADAPT
#######################################################################################################
# Path to your JSONL file
file_path = 'pathtoyourmodeloutputfile.jsonl'

# Define patterns and other words
pattern = r"true|false|ja|nein"
other_words = "was"
#other_words = ['ja', 'nein']
other_words_pattern = "|".join(re.escape(word) for word in other_words)
variable = 'suizidal'
default_pos = 'false'  # The positive default value
default = 'false'
#pattern = r"\b\d{1,2}\b|fehlt"
#pattern = r"\b\d{2}:\d{2}\b|fehlt"
#pattern = r"ja|nein"
# if there is a list with other words
# pattern = r"ja|nein" + "|".join(re.escape(word) for word in other_words)
#other_words = ['Sofia', 'RED', 'Catalyst']

csv_file_path = f'{variable}.csv'
#######################################################################################################

# Initialize lists to hold the data
reports = []
variables = []

# Read and process each line in the file

#%%
with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# Use the detected encoding
with open(file_path, 'r', encoding=result['encoding']) as file:
    for line in file:
        data = json.loads(line)

        # Extract data
        report = data.get('report', '')
        assistant_response = data.get('content', '')

        # Search for the primary pattern in the text
        primary_matches = re.findall(pattern, assistant_response, re.IGNORECASE)

        # Search for other words if no primary match
        if not primary_matches:
            other_matches = re.findall(other_words_pattern, assistant_response, re.IGNORECASE)
            matched_text = default_pos if other_matches else f'{default}'  # Default value if no match is found
        else:
            matched_text = ', '.join(primary_matches)  # Join primary matches with comma

        # Append to lists
        reports.append(report)
        variables.append(matched_text)

# #%%
# with open(file_path, 'rb') as rawdata:
#     result = chardet.detect(rawdata.read(100000))

# # Use the detected encoding
# with open(file_path, 'r', encoding=result['encoding']) as file:
#     for line in file:
#         data = json.loads(line)

#         # Extract data
#         report = data.get('report', '')
#         assistant_response = data.get('content', '')

#         # Search for the primary pattern in the text
#         primary_match = re.search(pattern, assistant_response, re.IGNORECASE)

#         # Initialize matched_text with a default value
#         matched_text = default_value

#         # If a primary match is found, use it
#         if primary_match is not None:
#             matched_text = primary_match.group(0)
#         else:
#             # Search for other words if no primary match
#             other_matches = re.findall(other_words_pattern, assistant_response, re.IGNORECASE)
#             if other_matches:
#                 matched_text = ', '.join(other_matches)

#         # Append to lists
#         reports.append(report)
#         variables.append(matched_text)

# Create DataFrame
df = pd.DataFrame({
    'report': reports,
    f'{variable}': variables,
})

#%%
# Display the DataFrame
print(df)

#%%
#save to csv

df.to_csv(f'{csv_file_path}')
