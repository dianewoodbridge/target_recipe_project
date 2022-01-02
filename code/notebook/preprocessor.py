
import pandas as pd 
import numpy as np
import string
import re
import ast

def preprocess(string_list):
    # Lowercase product titles
    string_list = [string.lower() for string in string_list]
    
    r = r'-'
    string_list = [re.sub(r, ' ', string) for string in string_list]
    
    # Remove qty information from product titles
    replace_expr = r'[0-9]+(.)?([0-9]+)?[\s]*(cans|can|boxes|box|bottles|bottle|\
                                              gallons|gallon|fl oz|oz|fl|gal|pk|\
                                              ct|ml|lbs|lb|qt|pt|ounce|0z|l|g)\b'
    string_list = [re.sub(replace_expr, '', string) for string in string_list]                                            
    
    # Remove punctuations from product titles
    regex = re.compile(r'[' + string.punctuation + '0-9\r\t\n]')
    string_list = [regex.sub("", string) for string in string_list]
    
    # Collapse multiple spaces into single space
    string_list = [re.sub("[\s]+", " ", string) for string in string_list]
    return string_list 
    
def preprocess_df(group10):
    # Remove all rows that do not have product title
    group10 = group10[~pd.isnull(group10['title'])].copy()

    columns = group10.columns
    if 'images' in  columns:
        # Change strings to lists
        group10['images'] = [ast.literal_eval(image_list) 
                        if not pd.isnull(image_list) 
                        else np.nan 
                        for image_list in group10['images']]
    if 'highlights' in columns:
        group10['highlights'] = [ast.literal_eval(image_list) 
                            if not pd.isnull(image_list) 
                            else np.nan 
                            for image_list in group10['highlights']]
    if 'specifications' in columns:
        group10['specifications'] = [ast.literal_eval(image_list) 
                            if not pd.isnull(image_list) 
                            else np.nan 
                            for image_list in group10['specifications']]
    if 'serving_info' in columns:
        group10['serving_info'] = [ast.literal_eval(image_list) 
                            if not pd.isnull(image_list) 
                            else np.nan 
                            for image_list in group10['serving_info']]
    if 'nutrition_info' in columns:
        group10['nutrition_info'] = [ast.literal_eval(image_list) 
                            if not pd.isnull(image_list) 
                            else np.nan 
                            for image_list in group10['nutrition_info']]

    # Preprocess product titles
    group10['title_modified'] = preprocess(group10['title'])

    # Stem product titles
    # group10['title_lower_stemmed']= group10['title_lower'].apply(stem_ingredient)
    return group10