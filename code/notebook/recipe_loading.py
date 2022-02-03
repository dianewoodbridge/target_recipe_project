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
    unit_regex = r" ounces | ounce | teaspoon | teaspoons | cups | cup | tablespoon | tablespoons | tbsp | tsp | can | lb | pound "
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
                    u = re.findall(unit_regex, rem,  flags=re.I)
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
                        re.split(unit_regex, rem, flags=re.I)[0].strip()
                        qty_list = re.findall('[0-9/]+', qty)

                        if len(qty_list) != 0:
                            dict_ingredients['unit'].append(u[0])
                            dict_ingredients['quantity'].append(qty_list[-1])
                            j = \
                            re.split(unit_regex, rem,  flags=re.I)[1].strip()
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
    unit_regex = r" ounces | ounce | teaspoon | teaspoons | cups | cup | tablespoon | tablespoons | tbsp | tsp | can | lb | pound "
    
    for lis in recipe[i]['ingredients']:
        for key, val in lis.items():
            ingredients_list.append(val)
            rem = re.sub("[\(\[].*?[\)\]]", "", val)
            rem = re.sub(' c. ', ' cup ', rem)
            rem = re.sub("[.]", "", rem)
            if rem != '':
                u = re.findall(unit_regex, rem,  flags=re.I)
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
                    qty = re.split(unit_regex, rem,  flags=re.I)[0].strip()
                    qty_list = re.findall('[0-9/]+', qty)

                    if len(qty_list) == 1:
                        dict_ingredients['unit'].append(u[0])
                        dict_ingredients['quantity'].append(qty_list[0])
                        j = re.split(unit_regex, rem,  flags=re.I)[1].strip()
                        dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                    elif len(qty_list) > 1:
                        
                        if re.findall(r'/', qty):
                            qt = qty_list[0] + '-' + qty_list[1]
                            dict_ingredients['quantity'].append(qt)
                            dict_ingredients['unit'].append(u[0])
                            j = re.split(unit_regex, rem,  flags=re.I)[1].split('or')[0]
                            dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                        elif len(qty_list[1])>1:
                            qt = qty_list[0] + '-'+ qty_list[1][0] + '/'+ qty_list[1][1]
                            dict_ingredients['quantity'].append(qt)
                            dict_ingredients['unit'].append(u[0])
                            j = re.split(unit_regex, rem,  flags=re.I)[1].split('or')[0]
                            dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                        else: 
                            dict_ingredients['quantity'].append(qty_list[0])
                            dict_ingredients['unit'].append('count')
                            j = ' '.join(i for i in qty.split(' ')[1:])
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
    pattern_1 = r'/'
    pattern_2 = r'-'
    if '/' in re.findall(pattern_1, utf) and '-' in re.findall(pattern_2, utf):
        first =  re.split(pattern_2, utf)
        d = re.split(pattern_1, first[1].strip())
        number =int(first[0].strip()) + int(d[0]) / int(d[1])
        return number    
    
    elif '/' in re.findall(pattern_1, utf):
        d = re.split(pattern_1, utf)
        number = int(d[0]) / int(d[1])
        return number

    return utf
