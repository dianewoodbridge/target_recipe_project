from preprocessor import *
import matplotlib.pyplot as plt

class DisplayProducts():
    def __init__(self, ranker, mapper):
        self.ranker = ranker
        self.mapper = mapper
    
    def display_products_recipe(self, recipe_ingredients):
        for ingredient in recipe_ingredients:
            self.display_products_ingredient(ingredient)
    
    def display_products_ingredient(self, ingredient):
        preprocessed_ingredient = preprocess([ingredient])
        tcin_list = self.ranker.rank_products_ingredient(preprocessed_ingredient)
        n_tcin = len(tcin_list)
        images = self.mapper.get_image_list(tcin_list)
        titles = self.mapper.get_title_list(tcin_list)
        
        print(ingredient)
        plt.figure(figsize=(20,10))
        columns = 3
        for i in range(n_tcin):
            plt.subplot(int(n_tcin / columns) + 1, columns, i + 1)
            plt.axis('off')
            plt.title(titles[i])
            if not isinstance(images[i], float):
                plt.imshow(images[i])
        plt.show()