from py4ami.wikimedia import WikidataLookup
from pathlib import Path
from pathlib import WindowsPath
path = WindowsPath('C:/Users/Roopa','hits_words.txt')
assert path.exists(),f"{path} exists"
wikidata_lookup = WikidataLookup()
file1 = open('extracted_words.txt', 'r')
lines = file1.readlines()
words_found = set()
with open(path, "w") as f:
    for i,line in enumerate(lines):
        if line.endswith("\n"):
            line = line[:-1]
        if line in words_found:
            continue
        words_found.add(line)
        print(line)
        if line.startswith("-"):
            print(f"skipping {line}")
            continue
        qitem0, desc, wikidata_hits = wikidata_lookup.lookup_wikidata(line)
        print("HITS", wikidata_hits)
        words = [line, qitem0, desc, str(wikidata_hits)]
        try:
            f.write(str(words) + "\n")
        except UnicodeEncodeError as e:
            print(f"cannot write",e)
        if wikidata_hits == 'None':
            pass
        else:
            continue
print(f"wrote {path}")