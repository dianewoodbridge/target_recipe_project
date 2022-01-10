from sentence_transformers import util

class TransformerRanker:
    def __init__(self, model, product_ids, max_rank=100):
        self.model = model
        self.max_rank = max_rank
        self.product_ids = product_ids
    
    def fit(self, documents):
        self.embeddings = self.model.encode(documents, 
                                            convert_to_tensor=True)
        
    def load_embeddings(self, embeddings):
        self.embeddings = embeddings
        
    def get_scores_ingredient(self, ingredient):
        ingredient_embedding = self.model.encode(ingredient, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(ingredient_embedding, self.embeddings)[0]
        product_score = dict(zip(self.product_ids, scores.numpy()))
        product_score = sorted(product_score.items(), 
                                key = lambda x: x[1], 
                                reverse=True)
        return product_score[0:self.max_rank]
        
    def get_scores_recipe(self, ingredient_list):
        recipe_scores = []
        for ingredient in ingredient_list:
            ingredient_scores = self.get_scores_ingredient(ingredient)
            recipe_scores.append(ingredient_scores)
        return recipe_scores
    
    def rank_products_ingredient(self, ingredient):
        product_scores = self.get_scores_ingredient(ingredient)
        return [product_score[0] for product_score in product_scores]
    
    def rank_products_recipe(self, ingredient_list):
        recipe_scores = self.get_scores_recipe(ingredient_list)
        return [[product_score[0] for product_score in product_scores] 
                for product_scores in recipe_scores]