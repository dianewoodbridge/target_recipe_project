import pandas as pd 
import numpy as np
import re
from preprocessor import *

class Qty_normal_map:
    def __init__(self, op_file_path, data):
        self.unit_abbreviation = { 'tbsp' : ['tbsp', "tablespoon","tablespoons"],
                      'tsp' : ['tsp', 'teaspoon', 'teaspoons'],
                     'ml' : ['ml', 'milliliter','milliliters'],
                     'cup' : ['cups','cup'],
                     'oz' : ['ounces','oz', 'ounce'] , 
                     'lb' : ['pound','lb','lbs','lbs.']
                        }
        self.op_file_path = op_file_path
        self.data = data
        
    #Normalizes quantity required
    def normalize_units(self, combined_ingredient_df):
        normalized_units = list()
        m_list=[]
        for ingredient in combined_ingredient_df.iterrows():
            unit = ingredient[1][1].strip()
            normalized_unit = ''
            for key, val in self.unit_abbreviation.items():
                if unit.lower() in val:
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
                    elif normalized_unit == 'lb':
                         m = 480
            if normalized_unit == '':
                normalized_unit = unit
                m = 0
            normalized_units.append(normalized_unit)
            m_list.append(m)
        combined_ingredient_df['quantity']= combined_ingredient_df['quantity'].astype(float)
        combined_ingredient_df['normalized_unit'] = normalized_units
        combined_ingredient_df['Volume_in_ml'] = combined_ingredient_df['quantity']*m_list
        
        return combined_ingredient_df
    def combine_qty(self, df):
        return df.groupby(by=['ingredient', 'normalized_unit'], as_index = False)\
                                      .agg({'quantity': 'sum', 'Volume_in_ml': 'sum'})


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
            elif row.normalized_unit=='lb':
                req_oz.append(row.quantity*16)
            elif 'cup' in row.standard_unit.strip():
                req_gm = (row.standard_weight_gm/225)*row.Volume_in_ml
                req_oz.append(req_gm/28.35)
            elif row.standard_unit.strip() in ['tbsp','tablespoon']:
                req_gm = (row.standard_weight_gm/15)*row.Volume_in_ml
                req_oz.append(req_gm/28.35)
            elif row.standard_unit.strip() in ['tsp','teaspoon']:
                req_gm = (row.standard_weight_gm/5)*row.Volume_in_ml
                req_oz.append(req_gm/28.35)
            elif row.normalized_unit == 'cup':
                 req_oz.append(row.quantity*8)
            elif row.normalized_unit == 'tsp':
                 req_oz.append(row.quantity*0.16667)
            elif row.normalized_unit == 'tbsp':
                 req_oz.append(row.quantity*0.5)
            else: req_oz.append(0)

        df['req_oz']=req_oz
        df['req_oz']=np.round(df['req_oz'], 3)
        return df

    def match_ranked_ingredients(self, ranked_match, final_df, recipe_ingredients):

        rslt_df = self.data[['title', 'tcin', 'short_desc','price','net_content_quantity_unit_of_measure', 'net_content_quantity_value', 'package_weight_unit_of_measure','package_weight']]
        final_rslt_df=pd.DataFrame()

        for i in range(len(ranked_match)):
            rslt_inter = rslt_df.loc[self.data['tcin'].isin(ranked_match[i])] 
            ing = recipe_ingredients[i]
            length = min(len(ranked_match[i]),9)
            for n in range(0,length):
                for j, row in rslt_inter.iterrows():
                    if row.tcin == ranked_match[i][n] :
                        rslt_inter.loc[j,'rank']=n+1
                        rslt_inter.loc[j,'ingredient']=ing
                        break
               
            rslt_inter_n=rslt_inter.sort_values('rank')[0:9] 
            final_rslt_df= pd.concat([final_rslt_df,rslt_inter_n], ignore_index=True)

        join_df = pd.merge(final_rslt_df, final_df, how = 'left', on = 'ingredient')
        return join_df

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
                        rec=0
                        if re.search(ele, row.ingredient, flags=re.I):
                            if df_count_to_weight['Unit'][i].strip() == 'oz':
                                rec = df_count_to_weight['Weight'][i]*row.quantity 
                                final_req = rec/row.package_weight
                                break
                            else:
                                rec = df_count_to_weight['Weight'][i]*16*row.quantity
                                final_req = rec/row.package_weight
                                break
                    if rec==0:
                        if row.normalized_unit == 'count' and row.quantity == 1:
                            final_req=1
                        else:
                            final_req=0
                           
                    recommended_qty.append(np.ceil(final_req))
                    product_qty.append(row.package_weight)
                elif re.search('pound', row.package_weight_unit_of_measure, flags=re.I):
                    for i, ele in enumerate(list_ing):
                        rec=0
                        if re.search(ele, row.ingredient, flags=re.I):
                            if df_count_to_weight['Unit'][i].strip() == 'oz':
                                rec = (df_count_to_weight['Weight'][i]/16)*row.quantity
                                final_req = rec/row.package_weight
                                break
                            else:
                                rec = df_count_to_weight['Weight'][i]*row.quantity
                                final_req = rec/row.package_weight
                                break
                    if rec==0:
                        if row.normalized_unit == 'count' and row.quantity == 1:
                            final_req=1
                        else:
                            final_req=0
 
                    recommended_qty.append(np.ceil(final_req))
                    product_qty.append(row.package_weight)
                else:    
                    recommended_qty.append(0)
                    product_qty.append(0)

        df['product_qty_oz_ct'] = product_qty
        df['recommended_qty'] = recommended_qty
 
        return df