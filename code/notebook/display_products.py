from itertools import product
from preprocessor import *
import matplotlib.pyplot as plt

class DisplayProducts():
    def __init__(self, ranker, mapper):
        self.ranker = ranker
        self.mapper = mapper
    
    def display_products_recipe(self, recipe_ingredients):
        for ingredient in recipe_ingredients:
            self.display_products_ingredient(ingredient)
    
    def display_products_ingredient(self, ingredient, n=3):
        preprocessed_ingredient = preprocess([ingredient])
        product_scores = self.ranker.get_scores_ingredient(preprocessed_ingredient, max_rank=n)     
        tcin_list = [product_score[0] for product_score in product_scores]
        scores_list = [product_score[1] for product_score in product_scores]
        n_tcin = len(tcin_list)
        images = self.mapper.get_image_list(tcin_list)
        titles = self.mapper.get_title_list(tcin_list)
        
        print(ingredient) 
        print(tcin_list)
        plt.figure(figsize=(20,10))
        columns = 3
        for i in range(n_tcin):
            plt.subplot(int(n_tcin / columns) + 1, columns, i + 1)
            plt.axis('off')
            plt.title(f'{titles[i]} ({scores_list[i]:.4f})')
            if not isinstance(images[i], float):
                plt.imshow(images[i])
        plt.show()

    def display_products_recipe_tcins(self, tcin_lists):
        for tcin_list in tcin_lists:
            self.display_products_ingredient_tcins(tcin_list)

    def display_products_ingredient_score(self, product_scores):
        tcin_list = [product_score[0] for product_score in product_scores]
        scores_list = [product_score[1] for product_score in product_scores]
        n_tcin = len(tcin_list)
        images = self.mapper.get_image_list(tcin_list)
        titles = self.mapper.get_title_list(tcin_list)
        
        print(tcin_list)
        plt.figure(figsize=(20,10))
        columns = 3
        for i in range(n_tcin):
            plt.subplot(int(n_tcin / columns) + 1, columns, i + 1)
            plt.axis('off')
            plt.title(f'{titles[i]} ({scores_list[i]:.4f})')
            if not isinstance(images[i], float):
                plt.imshow(images[i])
        plt.show()

    def display_products_ingredient_tcins(self, tcin_list):
        n_tcin = len(tcin_list)
        images = self.mapper.get_image_list(tcin_list)
        titles = self.mapper.get_title_list(tcin_list)
        
        print(tcin_list)
        plt.figure(figsize=(20,10))
        columns = 3
        for i in range(n_tcin):
            plt.subplot(int(n_tcin / columns) + 1, columns, i + 1)
            plt.axis('off')
            plt.title(titles[i])
            if not isinstance(images[i], float):
                plt.imshow(images[i])
        plt.show()

    def display_products_df(self, display_df, n=3):
        unique_ingredients = display_df['ingredient'].unique()
        for ingredient in unique_ingredients:
            tcin_list = display_df[display_df['ingredient'] == ingredient]['tcin'].values[0:n]
            recommended_qty_list = display_df[display_df['ingredient'] == ingredient]['recommended_qty'].values[0:n]
            price_list = display_df[display_df['ingredient'] == ingredient]['price'].values[0:n]

            n_tcin = len(tcin_list)
            images = self.mapper.get_image_list(tcin_list)
            titles = self.mapper.get_title_list(tcin_list)
            
            print(ingredient) 
            print(tcin_list)
            plt.figure(figsize=(20,10))
            columns = 3
            for i in range(n_tcin):
                ax = plt.subplot(int(n_tcin / columns) + 1, columns, i + 1)
                plt.axis('off')
                plt.title(titles[i])
                if not isinstance(images[i], float):
                    plt.imshow(images[i])
                ax.text(0.1, -0.1, f"Recommended Qty: {recommended_qty_list[i]: .0f}", size=12, ha="left", 
                        transform=ax.transAxes)
                ax.text(0.1, -0.17, f"Price: {price_list[i]: .2f}", size=12, ha="left", 
                        transform=ax.transAxes)
            plt.show()
    def display_products_df_kitchen_gadgets(self, display_df, n=3):
        unique_ingredients = display_df['cooking_tool'].unique()
        for ingredient in unique_ingredients:
            tcin_list = display_df[display_df['cooking_tool'] == ingredient]['tcin'].values[0:n]
            price_list = display_df[display_df['cooking_tool'] == ingredient]['price'].values[0:n]

            n_tcin = len(tcin_list)
            images = self.mapper.get_image_list(tcin_list)
            titles = self.mapper.get_title_list(tcin_list)
           
            plt.figure(figsize=(20,10))
 
            columns = 3
            for i in range(n_tcin):
                ax = plt.subplot(int(n_tcin / columns) + 1, columns, i + 1)
                plt.axis('off')
                plt.title(titles[i])
                if not isinstance(images[i], float):
                    plt.imshow(images[i])
                ax.text(0.1, -0.1, f"Product: {ingredient}", size=12, ha="left", 
                        transform=ax.transAxes)
                ax.text(0.1, -0.17, f"Price: {price_list[i]: .2f}", size=12, ha="left", 
                        transform=ax.transAxes)
            plt.show()