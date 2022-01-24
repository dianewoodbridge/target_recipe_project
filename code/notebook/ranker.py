from sentence_transformers import util

class TransformerRanker:
    def __init__(self, model, product_ids, max_rank=100,  clf=None):
        self.model = model
        self.max_rank = max_rank
        self.product_ids = product_ids
        self.clf = clf
    
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
        product_scores = dict(zip(self.product_ids, scores.numpy()))
        product_scores = sorted(product_scores.items(), 
                                key = lambda x: x[1], 
                                reverse=True)[0:100]
        if self.clf:
            tcin_list =  [product_score[0] for product_score in product_scores]
            tcin_list = self.clf.filter_by_class(ingredient, tcin_list)
            product_scores = [product_score for product_score in product_scores if product_score[0] in tcin_list]
        return product_scores[0:max_rank]
        
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
    def __init__(self, bi_model, cross_model, tcin_sentence_map, cross_rank=10, 
                 bi_rank=50):
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


from fse import IndexedList
from fse.models import SIF
from gensim.models import FastText
import pandas as pd
from preprocessor import *
class FastRanker:
    def __init__(self, embeddings, data, max_rank=100):
        self.embeddings = embeddings
        self.max_rank = max_rank
        self.data = data

    def fit(self, documents):
        tokenized_sentences = [head.split() for head in preprocess(documents)]
        self.indexed_docs = IndexedList(tokenized_sentences)
        self.data['processed'] = tokenized_sentences
        self.model = SIF(self.embeddings , workers=8)
        self.model.train(self.indexed_docs)
        
    def get_scores_ingredient(self, search_string, max_rank=10):
        if not max_rank:
            max_rank=self.max_rank
        process_search_str = search_string.split()
        matching_idx = []
        matched_data = self.model.sv.similar_by_sentence(sentence=process_search_str,
                                                        model=self.model, 
                                                        indexable=self.indexed_docs.items,
                                                        topn=max_rank)
        for match in matched_data:
            matching_idx.append((match[1], match[2]))

        result_df = pd.DataFrame(columns=['tcin', 'similarity'], index=range(max_rank))
        for i in range(max_rank):
            result_df['tcin'][i] = self.data.tcin.iloc[matching_idx[i][0]]
            result_df['similarity'][i] = matching_idx[i][1]
        return list(result_df.to_records(index=False))
        
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

class LabelEncoderWithNA():
    def fit(self, train, col):
        train[col] = train[col].astype('category').cat.as_ordered()
        self.encoder = train[col].cat.categories
    def transform(self, val, col):
        val[col] = pd.Categorical(val[col], categories=self.encoder, ordered=True)
        val[col] = val[col].cat.codes
    def fit_transform(self, train, col):
        self.fit(train, col)
        self.transform(train, col)

class Classifier():
    def __init__(self, model, pm, hier_column, threshold=8.9):
        self.hier_column = hier_column
        self.model = model
        self.pm = pm
        self.threshold=threshold
        self.le = LabelEncoderWithNA()
        self.le.fit_transform(self.pm.df.copy(), self.hier_column)

    def predict(self, x):
        return self.model.predict(x)

    def filter_by_class(self, ingredient, tcin_list):
        class_list = self.pm.get_column_list(tcin_list, 'class_name')
        scores = self.model.predict([ingredient])
        n = 1
        labels = [(self.le.encoder[score_argmax], score_max)
          for score_argmax, score_max 
          in zip(scores.argsort()[-n:][::-1], sorted(scores, reverse=True)[0:n])]
        if labels[0][1] > self.threshold:
           idx = [i for i, class_name in enumerate(class_list) if class_name == labels[0][0]]
           tcin_list = [tcin_list[i] for i in idx]
           print(f'Filtered {ingredient} for {self.hier_column}: {labels[0][0]}')
        return tcin_list


