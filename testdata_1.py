# This script generate the test data base on Flores200 for prompt instructed
# translation. It prompts the model to translate a sentence in one of the main
# languages in luxembourg (french, german, portuguese, english) to
# luxembourgish. This script uses only the test data, while the dev data can be
# used for instruction tuning. Jader Martins, 2024.

from datasets import load_dataset

target = {
    "Lëtzebuergesch": load_dataset("Muennighoff/flores200", "ltz_Latn"),
}

source = {
    "Portugisesch": load_dataset("Muennighoff/flores200", "por_Latn"),
    "Franséisch": load_dataset("Muennighoff/flores200", "fra_Latn"),
    "Däitsch": load_dataset("Muennighoff/flores200", "deu_Latn"),
    "Englesch": load_dataset("Muennighoff/flores200", "eng_Latn"),
}


prompt = "Translate the following sentences from {} to Lëtzebuergesch.\n"

for lang in source:
    for example in source[lang]["devtest"]["sentence"]:
        print(prompt.format(lang))
        for a, b in zip(source[lang]["dev"][:3]["sentence"],
                        target["Lëtzebuergesch"]["dev"][:3]["sentence"]):
            print(f"{lang}: {a}")
            print(f"Lëtzebuergesch: {b}")
            print()
        print(f"{lang}: {example}")
        print("Lëtzebuergesch: ", end="")
        exit()
