import requests
import json
import re
import pandas as pd
import numpy as np


def recipe_load_index(i, recipe):
    dict_ingredients = {'ingredient': [],
                        'unit': [],
                        'quantity': []
                        }
    title = recipe[i]['title']
    id = recipe[i]['id']
    print(f'Recipe: {title}')
    unit_regex = r" ounces | ounce | teaspoon | grams | g | teaspoons | cups | cup | tablespoon | tablespoons | tbsp | tsp | can | lb | pound | count | ml | pinch | pounds "
    
    for lis in recipe[i]['ingredients']:
        for key, val in lis.items():
   
            rem = re.sub("[\(\[].*?[\)\]]", "", val)
            rem = re.sub(' c. ', ' cup ', rem)
            rem = re.sub("[.]", "", rem)
            rem = re.sub("packages", "count", rem)
            rem = re.sub("cloves", "clove", rem)
            
            #multiple ingredients separated by 'or' - return first one wth unit
            if ' or ' in rem:
                    multi_ing = re.split(' or ', rem)
#                     print(multi_ing)
                    for j in range(len(multi_ing)):
                        u = re.findall(unit_regex, multi_ing[j],  flags=re.I)
                        if len(u)>0:
                            rem = multi_ing[j]
                            break
                            
                            
                            
            # If ingredient read
            if rem != '':
                u = re.findall(unit_regex, rem,  flags=re.I)
                #if no unit read
                
                if len(u) == 0:
                    qty = re.split(' ', rem)
                    qty_list = re.findall('[0-9/]+', qty[0])
                    if ('clove garlic' in rem) or ('garlic clove' in rem) :
                        dict_ingredients['unit'].append('cloves')
                        dict_ingredients['quantity'].append(qty_list[0])
                        dict_ingredients['ingredient'].append('garlic')
                        
                        
                    elif len(qty_list) != 0:
                        
                        dict_ingredients['quantity'].append(qty_list[-1])
                        dict_ingredients['unit'].append('count')
                        j = ' '.join(i for i in qty[1:])
                        dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                        
                    else:
                        dict_ingredients['quantity'].append(np.nan)
                        dict_ingredients['unit'].append('')
                        j = ' '.join(i for i in qty)
                        dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                # If unit read
                else:
                    qty = re.split(unit_regex, rem,  flags=re.I)[0].strip()
                    if 'to' in qty:
                        qty = re.split('to', qty,  flags=re.I)[1].strip()
                    elif 'about' in qty:
                        qty = re.split('about', qty,  flags=re.I)[1].strip()
                    qty_list = re.findall('[0-9/]+', qty)

                    # If single quantity parsed
                    if len(qty_list) == 1:
                        dict_ingredients['unit'].append(u[0])
                        dict_ingredients['quantity'].append(qty_list[0])
                        j = re.split(unit_regex, rem,  flags=re.I)[1].split(' or ')[0]
                        dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                    
                    # If multiple quantity values parsed  
                    elif len(qty_list) > 1:
                        # If quantity parsed in fractions
                        if re.findall(r'/', qty):
                            qt = qty_list[0] + '-' + qty_list[1]
                            dict_ingredients['quantity'].append(qt)
                            dict_ingredients['unit'].append(u[0])
                            j = re.split(unit_regex, rem,  flags=re.I)[1].split(' or ')[0]
                            dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                            
                        # If multiple quantities parsed are not in fractions 
                        elif len(qty_list[1])>1:
                            if u[0].strip() in ['grams','gram']:
                                qt = max(qty_list)
                                dict_ingredients['quantity'].append(qt)
                                dict_ingredients['unit'].append(u[0])
                                j = re.split(unit_regex, rem,  flags=re.I)[1].split(' or ')[0]
                                dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                            else:
                                qt = qty_list[0] + '-'+ qty_list[1][0] + '/'+ qty_list[1][1]
                                dict_ingredients['quantity'].append(qt)
                                dict_ingredients['unit'].append(u[0])
                                j = re.split(unit_regex, rem,  flags=re.I)[1].split(' or ')[0]
                                dict_ingredients['ingredient'].append(j.split(',')[0].strip())
                        elif int(qty_list[1])>1:
                            qt = max(qty_list)
                            dict_ingredients['quantity'].append(qt)
                            dict_ingredients['unit'].append(u[0])
                            j = re.split(unit_regex, rem,  flags=re.I)[1].split(' or ')[0]
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

    return id, dict_ingredients

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
    elif '-' in re.findall(pattern_2, utf):
        d = re.split(pattern_2, utf)
        number = int(d[0]) / int(d[1])
        return number
    return utf

def recipe_load_gadget(n, recipe):
    recipe_instr=[]
#     for i in range(s,n):
    title = recipe[n]['title']
    id = recipe[n]['id']
    print(title)
        
    for lis in recipe[n]['instructions']:
        for key, val in lis.items():   
            rem = re.sub("[\(\[].*?[\)\]]", "", val)
            if rem !='':
                recipe_instr.append(rem)
    return ' '.join(recipe_instr)
