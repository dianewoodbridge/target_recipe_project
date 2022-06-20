from rank_bm25 import BM25Okapi
from sentence_transformers import util
from pattern_search import *
from preprocessor import *
import torch
import joblib
from tqdm import tqdm

class Ranker: 
    '''
    Base Ranker class with common functionalities for TransformerRanker, CrossEncoderRanker
    and BM25Ranker classes
    '''       
    def get_scores_recipe(self, ingredient_list, max_rank=None, filtered_products_list=None):
        '''
        Input: List of ingredients required by a recipe
        Output: Matched products for each ingredient along with the relevance scores
        '''
        recipe_scores = []
        if filtered_products_list:
            for ingredient, filtered_products in zip(ingredient_list, filtered_products_list):
                ingredient_scores = self.get_scores_ingredient(ingredient, max_rank, 
                                                                filtered_products)
                recipe_scores.append(ingredient_scores)
        else:
            for ingredient in ingredient_list:
                ingredient_scores = self.get_scores_ingredient(ingredient, max_rank, 
                                                                filtered_products_list)
                recipe_scores.append(ingredient_scores)
        return recipe_scores

    def rank_products_ingredient(self, ingredient, max_rank=None, filtered_products=None):
        '''
        Returns only the matched product names for an ingredient and excludes the relevance 
        score
        '''
        product_scores = self.get_scores_ingredient(ingredient, max_rank, filtered_products)
        return [product_score[0] for product_score in product_scores]
    
    def rank_products_recipe(self, ingredient_list, max_rank=None, filtered_products=None):
        '''
        Returns only the matched product names for all the ingredients of a recipe
        and excludes the relevance score
        '''
        recipe_scores = self.get_scores_recipe(ingredient_list, max_rank, filtered_products)
        return [[product_score[0] for product_score in product_scores] 
                for product_scores in recipe_scores]

class TransformerRanker(Ranker):
    '''
    Level 1 Transformer Ranker for quick candidate retrieval
    '''
    def __init__(self, model, max_rank=100,  clf=None, filtered_products=None):
        self.model = model
        self.max_rank = max_rank
        self.clf = clf
        self.filtered_products = filtered_products
    
    def fit(self, documents):
        '''
        Encode the product texts using a transformer model
        '''
        self.embeddings = self.model.encode(documents, 
                                            convert_to_tensor=True)
        
    def load_embeddings(self, embeddings):
        '''
        Load already encoded transformer embeddings for product texts
        '''
        self.product_ids = pd.Series(embeddings['ids'])
        self.embeddings = embeddings['embeddings']
        
    def get_scores_ingredient(self, ingredient, max_rank=None, filtered_products=None):
        '''
        Input: An ingredient in a recipe
        Output: Matched products for the ingredient along with the relevance scores
        '''

        # Pre-filter any products if required
        if not filtered_products:
            filtered_products = self.filtered_products

        # Set the maximum number of matched products to be returned
        if not max_rank:
            max_rank=self.max_rank

        # Query expansion for improving matches
        multiple_nouns = get_noun_food(ingredient)
        if len(multiple_nouns) and multiple_nouns != ingredient:
            ingredient =  ingredient +  ' ' + multiple_nouns

        # Ingredient Embedding            
        ingredient_embedding = self.model.encode(ingredient, convert_to_tensor=True)

        # If there are filtered products already, search only in them
        if filtered_products:
            filtered_products_flag = self.product_ids.isin(filtered_products).values
            product_ids = self.product_ids[filtered_products_flag]
            embeddings = self.embeddings[filtered_products_flag]
        else:
            product_ids = self.product_ids
            embeddings = self.embeddings

        # Generate cosine similarity scores between ingredient embedding and all products
        scores = util.pytorch_cos_sim(ingredient_embedding, embeddings)[0]
        scores = scores.numpy()
        
        # Normalize scores
        min_doc_score = np.min(scores)
        max_doc_score = np.max(scores)
        scores = (scores - min_doc_score)/ (max_doc_score - min_doc_score)

        product_scores = dict(zip(product_ids, scores))

        # Reduce number of products to be classified for classification models (for speed)
        n_products_classification = min(len(product_scores), 100)

        # Sort products based on cosine similarity in descending order
        product_scores = sorted(product_scores.items(), 
                                key = lambda x: x[1], 
                                reverse=True)[0:n_products_classification]

        # If hierarchical classifiers are present then filter the products using the classifier
        # based on certain thresholds                     
        if self.clf:
            tcin_list =  [product_score[0] for product_score in product_scores]
            tcin_list_filtered = []
            for clf in self.clf:
                if clf.hier_column != 'subclass_name':
                    tcin_list_filtered += clf.filter_by_hier(ingredient, 
                                                             tcin_list, 
                                                             clf.hier_column)
                else:
                    tcin_list_subclass = clf.filter_by_hier(ingredient,
                                                             tcin_list, 
                                                             clf.hier_column)
                    if len(tcin_list_subclass) >= 3:
                        tcin_list_filtered = tcin_list_subclass
            if len(tcin_list_filtered) >= 3:
                tcin_list = tcin_list_filtered

            product_scores = [product_score 
                                for product_score 
                                in product_scores 
                                if product_score[0] in tcin_list]
        return product_scores[0:max_rank]

class CrossEncoderRanker(TransformerRanker):
    '''
    Similar to Transformer Ranker except that it does not compute cosine similarity. Instead,
    it gives a logit score for a pair of texts indicating their similairity
    '''
    def __init__(self, bi_model, cross_model, tcin_sentence_map, cross_rank=10, 
                 bi_rank=50, mapper=None, weights=False):
        self.bi_model = bi_model
        self.cross_model = cross_model
        self.cross_rank = cross_rank
        self.bi_rank = bi_rank
        self.tcin_sentence_map = tcin_sentence_map
        self.mapper = mapper
        self.weights = weights

    def get_scores_ingredient(self, ingredient, max_rank=None, filtered_products=None):
        if not max_rank:
            max_rank = self.cross_rank
        if isinstance(ingredient, list):
            ingredient = ingredient[0]

        # Get candidates quickly using a bi-encoder
        tcins = self.bi_model.rank_products_ingredient(ingredient, 
                                                        max_rank=self.bi_rank, 
                                                        filtered_products=filtered_products)                                          
        
        # Get product text for candidates
        sentences = []
        for tcin in tcins:
            sentences.append(self.tcin_sentence_map[self.tcin_sentence_map['tcin'] == tcin]['sentence'].values[0])
        
        # Generate pairs: (ingredient, product)
        pairs = [(ingredient, sentence.lower()) for sentence in sentences]
        
        # Compute logits
        scores = self.cross_model.predict(pairs)

        # Compute sigmoid
        scores = 1 / (1 + np.exp(-scores))

        # Boost scores for certain categories
        if self.weights == True:
            std = scores.std()
            for i, tcin in enumerate(tcins):
                if self.mapper.get_column_value(tcin, 'division_name') in ['PRODUCE/FLORAL', 'DRY GROCERY']:
                    scores[i] = scores[i] + 2 * std

        product_score = dict(zip(tcins, scores))
        product_score = sorted(product_score.items(), 
                                key = lambda x: x[1], 
                                reverse=True)
        return product_score[0:max_rank]

class BM25Ranker(Ranker):
    '''
    BM25Ranker for candidate retrieval
    '''
    def __init__(self, product_ids, max_rank=100, query_expansion=True):
        self.product_ids = product_ids
        self.max_rank = max_rank
        self.query_expansion = query_expansion

    def fit_corpus(self, articles, min_word_count = 1, op_path=None):
        corpus = []
        for article in tqdm(articles):
            corpus.append(tokenizer(article))

        # build a word count dictionary so we can remove words that appear only once
        word_count_dict = {}
        for text in corpus:
            for token in text:
                word_count = word_count_dict.get(token, 0) + 1
                word_count_dict[token] = word_count

        texts = [[token for token in text if word_count_dict[token] > min_word_count] for text in corpus]
        if op_path:
            joblib.dump(texts, op_path)
        return texts

    def fit(self, texts):
        self.model = BM25Okapi(texts)

    def get_scores_ingredient(self, ingredient, max_rank=None, filtered_list=None):
        if not max_rank:
            max_rank=self.max_rank
        if self.query_expansion:
            ingredient = query_expansion(ingredient)
        tokenized_query = tokenizer(ingredient)
        doc_scores = self.model.get_scores(tokenized_query)

        # Normalize
        min_doc_score = np.min(doc_scores)
        max_doc_score = np.max(doc_scores)
        doc_scores = (doc_scores - min_doc_score)/ (max_doc_score - min_doc_score)
        
        product_scores = dict(zip(self.product_ids, doc_scores))
        product_scores = sorted(product_scores.items(), 
                        key = lambda x: x[1], 
                        reverse=True)
        return product_scores[0:max_rank]         

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
    '''
    Filter products using a hierarchical classification model
    '''
    def __init__(self, model, pm, hier_column, threshold=8.9):
        self.hier_column = hier_column
        self.model = model
        self.pm = pm
        self.threshold=threshold
        self.le = LabelEncoderWithNA()
        self.le.fit_transform(self.pm.df.copy(), self.hier_column)

    def predict(self, x):
        return self.model.predict(x)

    def filter_by_hier(self, ingredient, tcin_list, hier_column):
        hier_column_list = self.pm.get_column_list(tcin_list, hier_column)
        scores = self.model.predict([ingredient])
        scores = torch.nn.functional.softmax(torch.tensor(scores)).numpy()
        n = 1
        labels = [(self.le.encoder[score_argmax], score_max)
          for score_argmax, score_max 
          in zip(scores.argsort()[-n:][::-1], sorted(scores, reverse=True)[0:n])]
        if labels[0][1] > self.threshold:
           idx = [i for i, hier_column_val 
                    in enumerate(hier_column_list) 
                    if hier_column_val == labels[0][0]]
           tcin_list = [tcin_list[i] for i in idx]
           print(f'Filtered {ingredient} for {self.hier_column}: {labels[0][0]}')
        else:
            tcin_list = []
        return tcin_list

class RankerPipeline:
    '''
    Use Rankers in a sequential stagewise manner with the RankerPipeline class
    '''
    def __init__(self, rankers, max_ranks):
        self.rankers = rankers
        self.max_ranks = max_ranks

    def get_scores_ingredient(self, ingredient):
        filtered_products = None
        for ranker, max_rank in zip(self.rankers, self.max_ranks):
            product_scores = ranker.get_scores_ingredient(ingredient, max_rank, 
                                                          filtered_products)
            filtered_products = [product_score[0] for product_score in product_scores]
        return product_scores

    def get_scores_recipe(self, ingredients):
        recipe_scores = []
        for ingredient in ingredients:
            ingredient_scores = self.get_scores_ingredient(ingredient)
            recipe_scores.append(ingredient_scores)
        return recipe_scores

    def rank_products_ingredient(self, ingredient):
        product_scores = self.get_scores_ingredient(ingredient)
        return [product_score[0] for product_score in product_scores]

    def rank_products_recipe(self, ingredients):
        recipe_scores = self.get_scores_recipe(ingredients)
        return [[product_score[0] for product_score in product_scores] 
                for product_scores in recipe_scores]

class RankerCombination:
    '''
    Combine Rankers using weights with the RankerCombination class
    '''
    def __init__(self, rankers, weights, max_rank = 10):
        self.rankers = rankers
        self.weights = weights
        self.max_rank = max_rank

    def get_scores_ingredient(self, ingredient, max_rank=None, filtered_products=None):
        if not max_rank:
            max_rank=self.max_rank

        for i, ranker in enumerate(self.rankers):
            ranker_scores = ranker.get_scores_ingredient(ingredient, max_rank,
                                                         filtered_products)
            if i == 0:
                df = pd.DataFrame(ranker_scores, columns=['tcin', 'score'])
                df['score'] = self.weights[i]*df['score']
            else:
                new_df =  pd.DataFrame(ranker_scores, columns=['tcin', 'score'])            
                df = pd.merge(df, new_df, how='outer', on='tcin').fillna(0)
                df['score'] = df['score_x'] + self.weights[i]*df['score_y']
                df.drop(columns=['score_x', 'score_y'], inplace=True)
        df = df.sort_values('score', ascending=False)[0: max_rank]
        return list(df.itertuples(index=False, name=None))

    def get_scores_recipe(self, ingredients):
        recipe_scores = []
        for ingredient in ingredients:
            ingredient_scores = self.get_scores_ingredient(ingredient)
            recipe_scores.append(ingredient_scores)
        return recipe_scores

    def rank_products_ingredient(self, ingredient):
        product_scores = self.get_scores_ingredient(ingredient)
        return [product_score[0] for product_score in product_scores]

    def rank_products_recipe(self, ingredients):
        recipe_scores = self.get_scores_recipe(ingredients)
        return [[product_score[0] for product_score in product_scores] 
                for product_scores in recipe_scores]


