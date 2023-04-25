"""Script to inspect the length of the files and make a descriptive summary. """

from pathlib import Path

import spacy

root = Path(__file__).parent.parent.parent / "data/308-docker.txt"


line_length = []

nlp = spacy.load("en_core_web_sm")
# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
# for token in doc:
#     print(token.text)

# TODO:
# BUSACR LIBRERÍA PARA VER SI ES UNA FECHA, PARA SIMPLIFICAR LA COMPARATIVA
# DE LOS FORMATOS

with open(root, "r") as f:
    docs = list(nlp.pipe(f.readlines()))
    print(len(docs))

    print("doc1: ", docs[0])
    print("doc2: ", docs[2])
    # for line in f.readline():
    #     doc = nlp(line)
    #     print(len(doc[1:]))

    # lines = f.readlines()
    # print(lines[:4])
    # print([len(l) for l in lines[:4]])
