# %%
import json
import requests
from tqdm import tqdm
import pandas as pd

#%%
output_json = "results_extractsuicid_nogram_llama2-70b_german_TF.jsonl"

grammar = r"""
root   ::= allrecords
value  ::= object | array | string | number | ("true" | "false" | "null") ws

allrecords ::= (
#  "{" ws "\"suizidal\":" ws (record)
    "{" ws "\"suizidal\":" ws ("null" | record)
  ws "}"
  ws
)

record ::= (
    "{" ws "\"Ist der Patient suizidal?\":" ws string
     "," ws "\"Ist der Patient suizidal?\":" ws ("true" | "false") ws 
    ws "}"
    ws
)

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n])?
"""
# prompt snippets
# You are a helpful medical assistant. You will be provided with a patient report. Strictly give the information from the report the user requests. Strictly keep with the content of the report, do not make anything up. You will be provided with definitions, that help answering the questions. Do not answer the questions with content of the definitions. 

prompt = """[INST] <<SYS>>
Sie sind ein aufmerksamer, medizinischer Assistent von OpenAI. Im Folgenden finden Sie eine psychiatrische Anamnese. Bitte beantworten Sie die Frage. 
<</SYS>>[/INST]
Beispiel:
Das ist die psychiatrische Anamnese:
Pat. wach, bewusstseinsklar, freundlich zugewandt und gesprächsbereit. Zeitlich unsicher (-1 Tag), örtlich, situativ und zur Person orientiert. Auffassung, Konzentration und Gedächtnis ungestört. Keine formalen oder inhaltlichen Denkstörungen, Störungen desIch-Erlebens oder derWahrnehmung. Stimmung ausgeglichen, Affekt situationsadäquat und schwingungsfähig. Keine Ängste oder Zwänge. Antrieb undPsychomotorik ungestört. Kein Lebensüberdruss oder Suizidgedanken. Keine akute Eigen- oder Fremdgefährdung. 
Ist der Patient suizidal? 
ASSISTANT: false. 
[INST]
Das ist die psychiatrische Anamnese:
{}
Ist der Patient suizidal? Antworte mit true oder false. 
[/INST]"""

# %%
df = pd.read_csv("merged_GT_3.csv", dtype=str) #.dropna(subset="content")
#df = df[:15]
#print(df.head(10))

#%%
try:
    with open(output_json, "r") as outjson:
        lines_so_far = sum(bool(line.strip()) for line in outjson)
except FileNotFoundError:
    lines_so_far = 0

with open(output_json, "a") as outjson:
    for report in tqdm(df.report.iloc[lines_so_far:]):
        #while True:
            #try:
        result = requests.post(
            url="http://127.0.0.1:8080/completion",
            json={
                "prompt": prompt.format("".join(report)),
                "n_predict": 2048,
                #"grammar": grammar,
                "temperature": 0.1,
            },
        )
        summary = result.json()
                #break
            #except json.decoder.JSONDecodeError:
            #    pass
        summary["report"] = report
        outjson.write(f"{json.dumps(summary, ensure_ascii=False)}\n")
        outjson.flush()
# %%