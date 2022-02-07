import pandas as pd
import numpy as np
import os
import re
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
import spacy

nlp = spacy.load('en_core_web_sm')

# Search functions
def flatten_dict(d):
    t = [v for (k,v) in d.items()]
    return [item for sublist in t for item in sublist]

def simple_length_ranker(product_matches):
    return sorted(product_matches, key=len)

def get_hypernym(ingredient):
    if ' ' in ingredient:
        ingredient = ingredient.replace(' ', '_')
    hypernym = ''
    try:
        synset = wn.synsets(ingredient)[0]
        hypernym = synset.hypernyms()[0].lemma_names()[0]
    except:
        pass
    if is_food(hypernym):
        return hypernym.replace('_', ' ')
    else:
        return ''

def get_hyponyms(ingredient):
    if ' ' in ingredient:
        ingredient = ingredient.replace(' ', '_')
    hyponym_list = []
    try:
        synsets = wn.synsets(ingredient)
        for synset in synsets:
            hyponyms = synset.hyponyms()
            for hyponym in hyponyms:
                hyponym_list += hyponym.lemma_names()
        hyponym_list = [h.replace('_', ' ') for h in hyponym_list if is_food(h)]
    except:
        pass
    return list(set(hyponym_list))

def get_synonyms(ingredient):
    if ' ' in ingredient:
        ingredient = ingredient.replace(' ', '_')
    synonym_list = []
    try:
        synsets = wn.synsets(ingredient)
        for synset in synsets:
            synonym_list += synset.lemma_names()
        synonym_list = [h.replace('_', ' ') for h in synonym_list if is_food(h)]
    except:
        pass
    return list(set(synonym_list))

def get_noun(ingredient):
    doc = nlp(ingredient)
    return " ".join([token.text for token in doc if token.pos_ == "NOUN"])

def is_food(ingredient):
    synsets = wn.synsets(ingredient)
    for synset in synsets:
        if 'food' in synset.lexname():
            return True

def stem_ingredient(ingredient):
    stemmer = PorterStemmer()
    stemmed = " ".join([stemmer.stem(w) for w in ingredient.split()])
    return stemmed

def get_noun_food(ingredient):
    noun = get_noun(ingredient)
    food_items = []
    if noun:
        if ' ' in noun:
            for s in noun.split():
                if is_food(s):
                    food_items.append(s)
            return " ".join(food_items)
    return noun

class PatternMatcher:
    def __init__(self, group10, k=10):
        self.group10 = group10
        self.k = k

    def search_stem(self, ingredient, k=10):
        stemmed_ingredient = stem_ingredient(ingredient)
        product_matches = []
        for product, product_stem in zip(self.group10['title_lower'], self.group10['title_lower_stemmed']):
            try:
                if bool(re.search(fr'\b{stemmed_ingredient}\b', product_stem)):
                    product_matches.append(product)
            except:
                pass
        product_matches = simple_length_ranker(set(product_matches))
        return product_matches[0:self.k]

    def search_exact(self, ingredient, k=10):
        product_matches = []
        for product in self.group10['title_lower']:
            try:
                if bool(re.search(fr'\b{ingredient}\b', product)):
                    product_matches.append(product)
            except:
                pass
        product_matches = simple_length_ranker(set(product_matches))
        return product_matches[0:self.k]

    def longest_match(self, ingredient, direction='backward'):
        product_matches = []
        split_ingredient = ingredient.split()
        while len(split_ingredient) > 1:
            if direction=='backward':
                split_ingredient = split_ingredient[:-1]
            elif direction=='forward':
                split_ingredient = split_ingredient[1:]
            ingredient = " ".join(split_ingredient)
            noun = get_noun(ingredient)
            if noun:
                if is_food(noun):
                    product_matches = self.search_exact(ingredient)
                    if len(product_matches) == 0:
                        product_matches = self.search_stem(ingredient)
                    if len(product_matches) > 0:
                        return product_matches
        return product_matches

    def search_noun(self, ingredient):
        product_matches = []
        noun = get_noun(ingredient)
        if noun:
            if is_food(noun):
                product_matches = self.search_exact(noun)
                if len(product_matches) == 0:
                    product_matches = self.search_stem(noun)
        return product_matches
            
    def search_noun_multiple(self, ingredient):
        noun = get_noun(ingredient)
        nouns = {}
        if noun:
            if ' ' in noun:
                for s in noun.split():
                    if is_food(s):
                        nouns[s] = self.search_exact(s)
                        if len(nouns[s]) == 0:
                            nouns[s] = self.search_stem(s)
                return nouns
        return nouns

    def search_hypernym(self, ingredient):
        product_matches = []
        hypernym = get_hypernym(ingredient)
        if hypernym:
            product_matches = self.search_exact(hypernym)
            if len(product_matches) == 0:
                product_matches = self.search_stem(hypernym)
        return product_matches

    def search_hyponyms(self, ingredient):
        combined_product_matches = []
        hyponym_list = get_hyponyms(ingredient)
        if len(hyponym_list) > 0:
            for hyponym in hyponym_list:
                product_matches = []
                product_matches = self.search_exact(hyponym)
                if len(product_matches) == 0:
                    product_matches = self.search_stem(hyponym)
                combined_product_matches += product_matches
        if len(combined_product_matches) > self.k: 
            return random.sample(combined_product_matches, self.k)
        return combined_product_matches