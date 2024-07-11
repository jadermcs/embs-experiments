# This task involves prediction if a sentence has identical or different
# meaning compared to other sentence.

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

np.random.seed(42)


tree = ET.parse('new_lod-art.xml')
root = tree.getroot()
data = []

for entry in root:
    lemma = entry.find("./lemma").text
    print(lemma)
    for meaning in entry.findall("./microStructure/grammaticalUnit/meaning"):
        for e in meaning.findall("./examples/example/text"):
            string = ""
            for i in e:
                text = i.text
                if text == "EGS":
                    continue
                string += text
                string += "" if text.endswith("'") else " "
            data.append({
                "lemma": lemma,
                "meaning": meaning.attrib["id"],
                "sentence": string})


def match_sentences(df):
    results = []
    for main_group, main_df in df.groupby('lemma'):
        all_sentences = set(main_df['sentence'])
        for sub_group, sub_df in main_df.groupby('meaning'):
            sub_sentences = set(sub_df['sentence'])
            outside_sentences = all_sentences - sub_sentences
            for sentence in sub_sentences:
                for outside_sentence in outside_sentences:
                    results.append({
                        'lemma': main_group,
                        'meaning': sub_group,
                        'sentence1': sentence,
                        'sentence2': outside_sentence
                    })
    return pd.DataFrame(results)


df = pd.DataFrame(data).sample(frac=1.)
matched_out = match_sentences(df).groupby("lemma").sample(1)
matched_out["label"] = "different"
words = set(matched_out.lemma.values)
print(matched_out)
matched_in = df.groupby(["lemma", "meaning"]).agg(lambda x: list(x)[:2]).reset_index()
matched_in = matched_in[[len(x) > 1 for x in matched_in.meaning.values.tolist()]]
matched_in = pd.concat([
    matched_in.drop(columns="sentence"),
    pd.DataFrame(
        matched_in['sentence'].values.tolist(),
        columns=["sentence1", "sentence2"])
    ], axis=1)
matched_in = matched_in[~matched_in.sentence2.isna() & matched_in.lemma.isin(words)]
matched_in = matched_in.groupby("lemma").sample(1)
matched_in["label"] = "identical"
print(matched_in)
df = pd.concat([matched_in, matched_out], ignore_index=True)
df = df.sample(frac=1.).drop(columns="meaning")
df.label = df.label.apply(
        lambda x: "richteg" if x == "identical" else "falsch")
df["prompt"] = "\"" + df['sentence1'] + "\"\r\n\r\n\"" + df['sentence2'] + "\""
df["answer"] = "Ist d'Bedeitung vun '" + df['lemma'] +\
        "' an deenen zwee Sätz identesch? " + df['label']
train = df.iloc[:-2000]
valid = df.iloc[-2000:-1000]
test = df.iloc[-1000:]

train.to_csv("data/dimension.train.csv", sep="\t", index=False)
valid.to_csv("data/dimension.valid.csv", sep="\t", index=False)
test.to_csv("data/dimension.test.csv", sep="\t", index=False)
