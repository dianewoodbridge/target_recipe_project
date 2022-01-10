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
        
    def get_scores_ingredient(self, ingredient, max_rank=None):
        if not max_rank:
            max_rank=self.max_rank

        ingredient_embedding = self.model.encode(ingredient, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(ingredient_embedding, self.embeddings)[0]
        product_score = dict(zip(self.product_ids, scores.numpy()))
        product_score = sorted(product_score.items(), 
                                key = lambda x: x[1], 
                                reverse=True)
        return product_score[0:max_rank]
        
    def get_scores_recipe(self, ingredient_list, max_rank=None):
        recipe_scores = []
        for ingredient in ingredient_list:
            ingredient_scores = self.get_scores_ingredient(ingredient, max_rank)
            recipe_scores.append(ingredient_scores)
        return recipe_scores

    def rank_products_ingredient(self, ingredient, max_rank=None):
        product_scores = self.get_scores_ingredient(ingredient, max_rank)
        return [product_score[0] for product_score in product_scores]
    
    def rank_products_recipe(self, ingredient_list, max_rank=None):
        recipe_scores = self.get_scores_recipe(ingredient_list, max_rank)
        return [[product_score[0] for product_score in product_scores] 
                for product_scores in recipe_scores]

    # Following code is not required
    def get_scores_ingredient_custom(self, ingredient, model, tokenizer):
        import torch
        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        #Tokenize sentences
        encoded_input = tokenizer(ingredient, padding=True, truncation=True, max_length=128, return_tensors='pt')
        #Compute token embeddings
        with torch.no_grad():
                model_output = model.encoder(
                    input_ids=encoded_input["input_ids"], 
                    attention_mask=encoded_input["attention_mask"], 
                    return_dict=True
                )
        #Perform pooling. In this case, mean pooling
        ingredient_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        scores = util.pytorch_cos_sim(ingredient_embedding, self.embeddings)[0]
        product_score = dict(zip(self.product_ids, scores.numpy()))
        product_score = sorted(product_score.items(), 
                                key = lambda x: x[1], 
                                reverse=True)
        return product_score[0:self.max_rank]


class CrossEncoderRanker(TransformerRanker):
    def __init__(self, bi_model, cross_model, tcin_sentence_map, cross_rank=10, bi_rank=50):
        self.bi_model = bi_model
        self.cross_model = cross_model
        self.cross_rank = cross_rank
        self.bi_rank = bi_rank
        self.mapper = tcin_sentence_map

    def get_scores_ingredient(self, ingredient, max_rank=None):
        if not max_rank:
            max_rank = self.cross_rank
        if isinstance(ingredient, list):
            ingredient = ingredient[0]
        tcins = self.bi_model.rank_products_ingredient(ingredient, max_rank=self.bi_rank)
        sentences = []
        for tcin in tcins:
            sentences.append(self.mapper[self.mapper['tcin'] == tcin]['sentence'].values[0])
        pairs = [(ingredient, sentence.lower()) for sentence in sentences]
        scores = self.cross_model.predict(pairs)
        product_score = dict(zip(tcins, scores))
        product_score = sorted(product_score.items(), 
                                key = lambda x: x[1], 
                                reverse=True)
        return product_score[0:max_rank]