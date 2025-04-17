import string
import re

extra_punct = "—"
extra_punct2 = "«"
extra_punct3 = "»"
extra_punct4 = "…"
punctuation = string.punctuation + extra_punct + extra_punct2 + extra_punct3 + extra_punct4

with open('cwiczenia1/pan-tadeusz.txt', 'r') as file:
    lines = file.readlines()

lines = [line.lower() for line in lines]

with open('cwiczenia1/tadeusz-clean.txt', 'w') as file:
    for line in lines:
        if not line.strip():
            continue
        line = line.replace('    ', '')
        translator = str.maketrans('', '', punctuation)
        clean = line.translate(translator)
        clean = re.sub(' +', ' ', clean)
        file.write(clean)
