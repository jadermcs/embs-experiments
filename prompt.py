import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from openai import OpenAI

client = OpenAI()

prompt = "You are presented with two sentences in Luxembourgish that both contain a specific word. Your task is to analyze how this word is used in each sentence and determine if its usage in the second sentence represents the same sense with respect to its use in the first sentence. Follow these steps to complete the task:\n\nStep 1. Describe the meaning of the word in the first sentence.\nStep 2. Describe the meaning of the word in the second sentence.\nStep 3. Based on the previous reasoning give your final answer with a single word: 'identical' or 'different.'"
prompt += "\n---"
prompt += "\nWord: Deckel"
prompt += "\n1. vergiess net, den Deckel vun der Këscht zouzeklappen!"
prompt += "\n2. d'Kand krut eng bei den Deckel"
prompt += "\nStep 1. In the first sentence, 'Deckel' is used to mean a cover of a recipient."
prompt += "\nStep 2. In the second sentence, 'Deckel' is used as a metaphor that for 'head.'"
prompt += "\nStep 3. different"
prompt += "\n---"
prompt += "\nWord: Zell"
prompt += "\n1. de Professer huet de Schüler den Opbau vun enger Zell erkläert"
prompt += "\n2. e Prisonéier huet sech a senger Zell erhaangen"
prompt += "\nStep 1. In sentence 1 it is used with the sense of a biological unity."
prompt += "\nStep 2. In sentence 2 is used with the sense of a unit in prison."
prompt += "\nStep 3. different"
prompt += "\n---"
prompt += "\nWord: Arm"
prompt += "\n1. kuck, du hues en Himmelsdéierchen um Aarm sëtzen!"
prompt += "\n2. de Sportler huet sech beim Training den Aarm gebrach"
prompt += "\nStep 1. In sentence 1 it is used with the sense of part of the body."
prompt += "\nStep 2. In sentence 2 it is used with the sense of member of a human body."
prompt += "\nStep 3. identical"
prompt += "\n---"

df = pd.read_csv("data/dimension.valid.csv", sep="\t")
preds = []
trues = []

for w, s1, s2, l in tqdm(df[["lemma", "sentence_1", "sentence_2", "label"]].values):
    tmp = prompt
    tmp += "\nWord: " + w
    tmp += "\n1. " + s1
    tmp += "\n2. " + s2
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": tmp}]
    )
    print(completion.choices[0].message.content)
    trues.append(l)
    preds.append(completion.choices[0].message.content.split()[-1].lower())

print(preds)
print(trues)
print(classification_report(preds, trues))
