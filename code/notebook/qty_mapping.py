import pandas as pd 
import numpy as np
import re
from preprocessor import *

class Qty_normal_map:
    def __init__(self, unit_abbreviation, op_file_path):
        self.unit_abbreviation = unit_abbreviation
        self.op_file_path = op_file_path
        
    #Normalizes quantity required
    def normalize_units(self, combined_ingredient_df):
        normalized_units = list()
        m_list=[]
        for ingredient in combined_ingredient_df.iterrows():
            unit = ingredient[1][1].strip()
            normalized_unit = ''
            for key, val in self.unit_abbreviation.items():
                if unit in val:
                    normalized_unit = key
                    if normalized_unit == 'cup':
                         m = 225
                    elif normalized_unit == 'tsp':
                         m = 5
                    elif normalized_unit == 'tbsp':
                         m = 15
                    elif normalized_unit == 'ml':
                         m = 1
                    elif normalized_unit == 'oz':
                         m = 30
            if normalized_unit == '':
                normalized_unit = unit
                m = 0
            normalized_units.append(normalized_unit)
            m_list.append(m)
        combined_ingredient_df['quantity']= combined_ingredient_df['quantity'].astype(float)
        combined_ingredient_df['normalized_unit'] = normalized_units
        combined_ingredient_df['Volume_in_ml'] = combined_ingredient_df['quantity']*m_list
        
        return combined_ingredient_df
    
    def search_density(self, ingredient):
        df = pd.read_csv(self.op_file_path)
        df['standard_unit'] = np.where(df['standard_unit'].isna(), '', df['standard_unit'])
        ing = (df['ingredient']).tolist()
        for n,i in enumerate(ing):
            if ingredient.lower() in i.lower():
                unit = re.sub("[,]", "", df.iloc[n,3])
                return (df.iloc[n,1], df.iloc[n,2] , unit)    
        return (None, None, '')
    
    def update_density(self, combined_ingredient_df):
        combined_ingredient_df['tuple'] = combined_ingredient_df['ingredient'].apply(self.search_density)
        combined_ingredient_df['standard_vol'],combined_ingredient_df['standard_weight_gm'],\
        combined_ingredient_df['standard_unit'] = combined_ingredient_df.tuple.str
        combined_ingredient_df = combined_ingredient_df.drop(columns='tuple')
        return combined_ingredient_df
    
    def req_oz_recipe(self, combined_ingredient_df):
        df = self.update_density(combined_ingredient_df)
        req_oz=[]

        for index, row in df.iterrows():
            if row.normalized_unit=='oz':
                req_oz.append(row.quantity)
            elif row.normalized_unit=='pound':
                req_oz.append(row.quantity*16)
            elif 'cup' in row.standard_unit.strip():
                req_gm = (row.standard_weight_gm/225)*row.Volume_in_ml
                req_oz.append(req_gm/28.35)
            elif 'tbsp' in row.standard_unit.strip() or 'tablespoon' in row.standard_unit.strip():
                req_gm = (row.standard_weight_gm/15)*row.Volume_in_ml
                req_oz.append(req_gm/28.35)
            elif 'tsp' in row.standard_unit.strip() or 'teaspoon' in row.standard_unit.strip():
                req_gm = (row.standard_weight_gm/5)*row.Volume_in_ml
                req_oz.append(req_gm/28.35)
            elif row.normalized_unit == 'cup':
                 req_oz.append(row.quantity*8)
            else: req_oz.append(0)

        df['req_oz']=req_oz
        df['req_oz']=np.round(df['req_oz'], 3)
        return df
    
    def read_count_to_weight(self):
        df = pd.read_csv('../../data/count_to_weight.csv')
        df['Weight']=df['Imperial'].apply(lambda x: float(x.split(' ')[0]))
        df['Unit']=df['Imperial'].apply(lambda x: x.split(' ')[1])
        df['Ingredient']=df['Ingredient'].apply(lambda x: x.strip())
        return df
    
    def recommended_quantity(self, df):
        recommended_qty=[]
        product_qty=[]
        
        
        df['package_weight']=df['package_weight'].apply(float)
        df['net_content_quantity_value']=df['net_content_quantity_value'].apply(float)
        
        df_count_to_weight = self.read_count_to_weight()
        list_ing = preprocess(df_count_to_weight['Ingredient'].tolist())
        
        for index, row in df.iterrows():
            if row.normalized_unit == '':
                recommended_qty.append(1)
                product_qty.append(row.package_weight)
                
                
            elif row.req_oz > 0 :

                if re.search('pound', row.package_weight_unit_of_measure, flags=re.I):
                    pack_oz = row.package_weight * 16
                    rec = row.req_oz/pack_oz
                    recommended_qty.append(np.ceil(rec))
                    product_qty.append(pack_oz)
                elif re.search('ounce', row.package_weight_unit_of_measure, flags=re.I):
                    rec = row.req_oz/row.package_weight
                    recommended_qty.append(np.ceil(rec))
                    product_qty.append(row.package_weight)
            else:
                if re.search('dozen', row.net_content_quantity_unit_of_measure, flags=re.I):
                    rec = row.quantity/(12*row.net_content_quantity_value)
                    recommended_qty.append(np.ceil(rec))
                    product_qty.append(12*row.net_content_quantity_value)
                elif re.search('count', row.net_content_quantity_unit_of_measure, flags=re.I):
                    rec = row.quantity
                    recommended_qty.append(np.ceil(rec))
                    product_qty.append(row.package_weight)
                elif re.search('each', row.net_content_quantity_unit_of_measure, flags=re.I):
                    rec = row.quantity
                    recommended_qty.append(np.ceil(rec))
                    product_qty.append(row.package_weight)
                elif re.search('ounce', row.package_weight_unit_of_measure, flags=re.I):
                    for i, ele in enumerate(list_ing):
                        if re.search(ele, row.ingredient, flags=re.I):
                            if df_count_to_weight['Unit'][i].strip() == 'oz':
                                rec = df_count_to_weight['Weight'][i]*row.quantity                      
                            else:
                                rec = df_count_to_weight['Weight'][i]*16*row.quantity
                    final_req = rec/row.package_weight
                    recommended_qty.append(np.ceil(final_req))
                    product_qty.append(row.net_content_quantity_value)
                elif re.search('pound', row.package_weight_unit_of_measure, flags=re.I):
                    for i, ele in enumerate(list_ing):
                        if re.search(ele, row.ingredient, flags=re.I):
                            if df_count_to_weight['Unit'][i].strip() == 'oz':
                                rec = (df_count_to_weight['Weight'][i]/16)*row.quantity                      
                            else:
                                rec = df_count_to_weight['Weight'][i]*row.quantity
                    final_req = rec/row.package_weight
                    recommended_qty.append(np.ceil(final_req))
                    product_qty.append(row.net_content_quantity_value)
                else:    
                    recommended_qty.append(0)
                    product_qty.append(0)
                    
        print(recommended_qty)
        df['product_qty_oz_ct'] = product_qty
        df['recommended_qty'] = recommended_qty
 
        return df