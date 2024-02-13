# %%
import json
import requests
from tqdm import tqdm
import pandas as pd

#%%
output_json = "youroutputfilename.jsonl"

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
df = pd.read_csv("yourinputfilewithreports.csv", dtype=str) #.dropna(subset="content")
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
