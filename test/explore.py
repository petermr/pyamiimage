from collections import defaultdict
import pprint
import json
taxonomy = None
def tree():
    return defaultdict(tree)

def test_colours():
    print("colours")

    tt = tree()
    tt['house']['car']['red']['hubcap'] = 1950

    s = [('red', 1), ('blue', 2), ('red', 3), ('blue', 4), ('red', 1), ('blue', 4)]
    d = defaultdict(set)
    for k, v in s:
        d[k].add(v)

    print("==sorted==\n", sorted(d.items()))


# https://gist.github.com/hrldcpr/2012250
def make_taxonomy():
    taxonomy = tree()
    taxonomy['Animalia']['Chordata']['Mammalia']['Carnivora']['Felidae']['Felis']['cat']
    taxonomy['Animalia']['Chordata']['Mammalia']['Carnivora']['Felidae']['Panthera']['lion']
    taxonomy['Animalia']['Chordata']['Mammalia']['Carnivora']['Canidae']['Canis']['dog']
    taxonomy['Animalia']['Chordata']['Mammalia']['Carnivora']['Canidae']['Canis']['coyote']
    taxonomy['Plantae']['Solanales']['Solanaceae']['Solanum']['tomato']
    taxonomy['Plantae']['Solanales']['Solanaceae']['Solanum']['potato']
    taxonomy['Plantae']['Solanales']['Convolvulaceae']['Ipomoea']['sweet potato']
    print(json.dumps(taxonomy, indent=4))
    del(taxonomy['Animalia'])
    print("del:", json.dumps(taxonomy, indent=4))
    return taxonomy

def dicts(t):
    return {k: dicts(t[k]) for k in t}

def addx(t, path):
    for node in path:
        print(">> ",t," > ",node)
        t = t[node]
    return t

def test_taxonomy():
    taxonomy = make_taxonomy()
    print("\ntaxonomy\n", taxonomy)
    ttt = dicts(taxonomy)
    print("\n>dicts>\n", ttt)

def test_taxonomy1():
    taxonomy = make_taxonomy()
    print("\n>taxonomy1>\n", taxonomy)
    tax1 = addx(taxonomy,
        'Animalia>Chordata>Mammalia>Cetacea>Balaenopteridae>Balaenoptera>blue whale'.split('>'))
    print("\n>tax1>\n....", tax1,"\n>taxonomy2>\n", taxonomy)
    taxonomy["test"] = "test"
    print("\n>taxonomy4>\n", taxonomy["test"])




"""
{'Animalia': {'Chordata': {'Mammalia': {'Carnivora': {'Canidae': {'Canis': {'coyote': {},
                                                                            'dog': {}}},
                                                      'Felidae': {'Felis': {'cat': {}},
                                                                  'Panthera': {'lion': {}}}}}}},
 'Plantae': {'Solanales': {'Convolvulaceae': {'Ipomoea': {'sweet potato': {}}},
                           'Solanaceae': {'Solanum': {'potato': {},
                                                      'tomato': {}}}}}}
"""


"""
{'Animalia': {'Chordata': {'Mammalia': {'Carnivora': {'Canidae': {'Canis': {'coyote': {},
                                                                            'dog': {}}},
                                                      'Felidae': {'Felis': {'cat': {}},
                                                                  'Panthera': {'lion': {}}}},
                                        'Cetacea': {'Balaenopteridae': {'Balaenoptera': {'blue whale': {}}}}}}},
 'Plantae': {'Solanales': {'Convolvulaceae': {'Ipomoea': {'sweet potato': {}}},
                           'Solanaceae': {'Solanum': {'potato': {},
                                                      'tomato': {}}}}}}
"""