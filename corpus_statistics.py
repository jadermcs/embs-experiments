import os
import re
import json
import string
import spacy
import stop_words
from spacy.language import Language
from collections import defaultdict
import xml.etree.ElementTree as ET


data = []
tokens = 0
freqs = {}


with open("lb_lemma_lookup.json") as fin:
    LUXEMBOURGISH_LEMMATIZER = json.load(fin)


def remove_punct(s):
    s = s.translate(str.maketrans('', '', string.digits))
    return s.translate(str.maketrans('', '', string.punctuation))


# Define the lemmatizer function
@Language.component("luxembourgish_lemmatizer")
def custom_lemmatizer(doc):
    for token in doc:
        lemma = LUXEMBOURGISH_LEMMATIZER.get(token.text, token.text)
        token.lemma_ = lemma
    return doc


# Register the custom lemmatizer
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('luxembourgish_lemmatizer', last=True)
nlp.max_length = 10_000_000


for path, subdirs, files in os.walk("../LOD-Corpus/Texter/ANER_TEXTER"):
    for name in files:
        fpath = os.path.join(path, name)
        print(fpath)
        if fpath.endswith("xml"):
            root = ET.parse(fpath).getroot()
            date = root.find("./cesHeader/fileDesc/sourceDesc/biblStruct/monogr/imprint/pubDate").text
            try:
                content = root.find("./text/body/p").text
            except Exception as e:
                print(e)
                print("Failed to:", fpath)
                continue
        else:
            date = re.search("[0-9]{4}", name)
            if date and 1900 < int(date.group()) < 2024:
                date = date.group()
            else:
                date = "null"
            with open(fpath) as fin:
                content = fin.read()
        print(date)
        tokens += len(content.split())
        for token in nlp(content):
            lemma = remove_punct(token.lemma_).strip()
            if lemma in stop_words.STOP_WORDS:
                continue
            if lemma:
                if lemma not in freqs:
                    freqs[lemma] = defaultdict(int)
                freqs[lemma][date] += 1

with open("freqs.json", "w+") as fout:
    json.dump(freqs, fout, indent=4)

print("Total tokens:", tokens)
