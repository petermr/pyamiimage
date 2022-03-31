from py4ami.wikimedia import WikidataLookup
from collections import Counter
from pathlib import Path
import pandas as pd
wikidata_lookup = WikidataLookup()
file1 = open('extracted_words.txt', 'r')
lines = file1.readlines()
words_found = set()
dict_wikidata_words= {}
word = None
word_counter = Counter()
print("iterating through words")
for i,line in enumerate(lines[:]):
    print(f"line={line}")
    if line.endswith("\n"):
        line = line[:-1]
    word_counter[line]+=1
    if word_counter[line]>1:
        print(f"counter{line}")
        continue
    words_found.add(line)
    print(line)
    if line.startswith("-"):
        #skip teserract misread of Greek characters
        print(f"skipping {line}")
        continue
    qitem0, desc, wikidata_hits = wikidata_lookup.lookup_wikidata(line)
    wikidata_words = {'word':line, 'qitem0': qitem0, 'description': desc, 'wikidata_hits':wikidata_hits}
    print(f"wikidata={wikidata_words}")
    word = line
    dict_wikidata_words[word] = wikidata_words
print(dict_wikidata_words)
print(f"Counter {word_counter}")
import json
with open('dictionary_words_1.json', 'w') as fp:
    json.dump(dict_wikidata_words, fp, indent=4)

#import json
#with open('/content/dictionary_words') as f:
#data = json.load(f)
