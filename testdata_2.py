# This task involves prediction if a sentence has identical or different
# meaning compared to other sentence.
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

np.random.seed(42)

tree = ET.parse('new_lod-art.xml')
words = set()
root = tree.getroot()
data = []

for entry in root:
    lemma = entry.find("./lemma").text
    for meaning in entry.findall("./microStructure/grammaticalUnit/meaning"):
        for e in meaning.findall("./examples/example/text"):
            meaning_txt = meaning.attrib["id"]
            string = ""
            for i in e:
                text = i.text
                if text == "EGS":
                    meaning_txt += "_EGS"
                    continue
                string += text
                string += "" if text.endswith("'") else " "
            words.add(lemma)
            data.append({
                "lemma": lemma,
                "meaning": meaning_txt,
                "sentence": string})

with open("words_lod.txt", "w+") as fout:
    for w in words:
        fout.write(w+"\n")

df = pd.DataFrame(data).sample(frac=1.)
df = pd.merge(df, df, on="lemma", suffixes=("_1", "_2"))
df = df[df.sentence_1 != df.sentence_2]
df.to_csv("data.csv", sep="\t", index=False)
df["label"] = df.apply(
    lambda x: "identical" if x["meaning_1"] == x["meaning_2"] else "different",
    axis=1)
df = df.drop(columns=["meaning_1", "meaning_2"])
df["answer"] = df.label.apply(
        lambda x: "richteg" if x == "identical" else "falsch")
df["prompt"] = "Ass d'Bedeitung vun '" + df['lemma'] +\
        "' an deenen zwee Sätz identesch? saz1: " + df['sentence_1'] +\
        " saz2: " + df['sentence_2']
test_words = np.random.choice(df.lemma.unique(), 55)
train = df[~df.lemma.isin(test_words)]
valid_words = np.random.choice(train.lemma.unique(), 55)
train = train[~train.lemma.isin(valid_words)]
print("Train size:", train.shape[0])
valid = df[df.lemma.isin(valid_words)]
print("Validation size:", valid.shape[0])
test = df[df.lemma.isin(test_words)]
print("Test size:", test.shape[0])

df.to_csv("data/dimension.complete.csv", sep="\t", index=False)
train.to_csv("data/dimension.train.csv", sep="\t", index=False)
valid.to_csv("data/dimension.valid.csv", sep="\t", index=False)
test.to_csv("data/dimension.test.csv", sep="\t", index=False)
