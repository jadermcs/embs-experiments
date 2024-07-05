# This task involves prediction if a word has identical or different meaning 
# compared to other word.

import xml.etree.ElementTree as ET
import pandas as pd


tree = ET.parse('new_lod-art.xml')
root = tree.getroot()
data = []

for entry in root:
    for meaning in entry.findall("./microStructure/grammaticalUnit/meaning"):
        for e in meaning.findall("./examples/example/text"):
            string = ""
            for i in e:
                text = i.text
                if text == "EGS":
                    continue
                string += text
                string += "" if text.endswith("'") else " "
            print(string)
            data.append({
                "entry": entry.attrib["id"],
                "meaning": meaning.attrib["id"],
                "sentence": string})

df = pd.DataFrame(data)
print(df.head())
print(df.shape)
print(df.entry.nunique())
