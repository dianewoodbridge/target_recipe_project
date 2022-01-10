import requests
import json
import re
import pandas as pd
import numpy as np


def recipe_load(n, recipe):
    dict_ingredients = {'ingredient': [],
                        'unit': [],
                        'quantity': []
                        }
    ingredients_list = []
    for i in range(0, n):
        title = recipe[i]['title']
        id = recipe[i]['id']

        for lis in recipe[i]['ingredients']:
            for key, val in lis.items():
                ingredients_list.append(val)
                rem = re.sub("[\(\[].*?[\)\]]", "", val)
                rem = re.sub(' c. ', ' cup ', rem)
                rem = re.sub("[.]", "", rem)
                if rem != '':
                    u = re.findall(r" ounces | ounce | teaspoon | cups | cup | tablespoon | tbsp | tsp | can ", rem)
                    if len(u) == 0:
                        qty = re.split(' ', rem)
                        qty_list = re.findall('[0-9/]+', qty[0])
                        if len(qty_list) != 0:

                            dict_ingredients['quantity'].append(qty_list[-1])
                            dict_ingredients['unit'].append('count')
                            j = ' '.join(i for i in qty[1:])
                            dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                        else:
                            dict_ingredients['quantity'].append(np.nan)
                            dict_ingredients['unit'].append('')
                            j = ' '.join(i for i in qty)
                            dict_ingredients['ingredient'].append(j.split(',')[0].strip())

                    else:
                        qty = \
                        re.split(r" ounces | ounce | teaspoon | cups | cup | tablespoon | tbsp | tsp | can ", rem)[
                            0].strip()
                        qty_list = re.findall('[0-9/]+', qty)

                        if len(qty_list) != 0:
                            dict_ingredients['unit'].append(u[0])
                            dict_ingredients['quantity'].append(qty_list[-1])
                            j = \
                            re.split(r"ounces | ounce | teaspoon | cups | cup | tablespoon | tbsp | tsp | can ", rem)[
                                1].strip()
                            dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                        else:
                            dict_ingredients['quantity'].append(np.nan)
                            dict_ingredients['unit'].append('')
                            j = ' '.join(i for i in qty)
                            dict_ingredients['ingredient'].append(j.split(',')[0].strip())

    return dict_ingredients

def recipe_load_index(i, recipe):
    dict_ingredients = {'ingredient': [],
                        'unit': [],
                        'quantity': []
                        }
    ingredients_list = []
    title = recipe[i]['title']
    id = recipe[i]['id']
    print(f'Recipe: {title}')

    for lis in recipe[i]['ingredients']:
        for key, val in lis.items():
            ingredients_list.append(val)
            rem = re.sub("[\(\[].*?[\)\]]", "", val)
            rem = re.sub(' c. ', ' cup ', rem)
            rem = re.sub("[.]", "", rem)
            if rem != '':
                u = re.findall(r" ounces | ounce | teaspoon | cups | cup | tablespoon | tbsp | tsp | can ", rem)
                if len(u) == 0:
                    qty = re.split(' ', rem)
                    qty_list = re.findall('[0-9/]+', qty[0])
                    if len(qty_list) != 0:

                        dict_ingredients['quantity'].append(qty_list[-1])
                        dict_ingredients['unit'].append('count')
                        j = ' '.join(i for i in qty[1:])
                        dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                    else:
                        dict_ingredients['quantity'].append(np.nan)
                        dict_ingredients['unit'].append('')
                        j = ' '.join(i for i in qty)
                        dict_ingredients['ingredient'].append(j.split(',')[0].strip())

                else:
                    qty = \
                    re.split(r" ounces | ounce | teaspoon | cups | cup | tablespoon | tbsp | tsp | can ", rem)[
                        0].strip()
                    qty_list = re.findall('[0-9/]+', qty)

                    if len(qty_list) != 0:
                        dict_ingredients['unit'].append(u[0])
                        dict_ingredients['quantity'].append(qty_list[-1])
                        j = \
                        re.split(r"ounces | ounce | teaspoon | cups | cup | tablespoon | tbsp | tsp | can ", rem)[
                            1].strip()
                        dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                    else:
                        dict_ingredients['quantity'].append(np.nan)
                        dict_ingredients['unit'].append('')
                        j = ' '.join(i for i in qty)
                        dict_ingredients['ingredient'].append(j.split(',')[0].strip())

    return dict_ingredients

def convert_fraction(utf):
    if utf is np.nan:
        return utf
    pattern = r'/'
    if '/' in re.findall(pattern, utf):
        d = re.split(pattern, utf)
        number = int(d[0]) / int(d[1])
        return number

    return utf
