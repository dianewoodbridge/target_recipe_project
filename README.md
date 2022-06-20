# target_recipe_project

## API Functionalities & Requirements:
Descriptions are provided in a brief and concise manner. For more details, please refer to the code file comments. The API has the following classes and functionalities that are implemented in the associated python files: 

1. **pattern_search.py**  
Contains the PatternMatcher class and associated functions for pattern matching. The functions associated include searching for products using exact string matching, stemmed string matching, searching for hypernyms, hyponyms, and food nouns.  
*Import Requirements: re, nltk, spacy*

2. **preprocessor.py**  
Contains functions for preprocessing ingredient and product query text. Some of the preprocessing functionalities that could be applied include lowercasing, stemming, tokenization, removing punctuations, stop words, quantities, brand information. There is also a query expansion functionality that can expand an ingredient query with the hypernyms, hyopnyms, synonyms, and food nouns that are identified by the WordNet lexical database.  
*Import Requirements: pattern_search, pandas, numpy, string, ast, sklearn*

3. **ranker.py**    
Contains the classes Ranker, TransformerRanker, CrossEncoderRanker, BM25Ranker, Classifier, RankerCombination, and RankerPipeline.
    - Ranker: Base Ranker class with common functionalities for TransformerRanker, CrossEncoderRanker and BM25Ranker classes. 
    - TransformerRanker:  This is the level 1 transformer ranker for quick candidate retrieval
    - CrossEncoderRanker: This is the level 2 transformer ranker for re-ranking. It takes in the level 1 transformer ranker as an argument for quick candidate retrieval after which it re-ranks using a cross-encoder. The cross-encoder gives a logit score for a pair of texts indicating their similairity.
    - BM25Ranker: This ranker can be used as a level 1 ranker for quick candidate retrieval. 
    - Classifier: This is the hierarchical classification class that contains functionalities for filtering products based on the product category that an ingredient most likely belongs to. 
    - RankerPipeline: Provides functionality to use different rankers (BM25Ranker, TransformerRanker, CrossEncoderRanker) in a sequential stagewise manner.
    - RankerCombination: Provides functionality to combine rankers using an array of weights corresponding to the rankers.    

*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Import Requirements: preprocessor, rank_bm25, sentence_transformers, torch, joblib*

4. **mapper.py**    
Contains the Mapper class that contains functions for mapping from a product_id to an image, title, or any other dataframe column. 
*Import Requirements: PIL, requests, io, numpy*

5. **display_products.py**  
Contains the DisplayProducts class that can display the most relevant products for a recipe or ingredient based on a ranker. It can also display products based on a product_id or list of product_ids. The products are displayed along with the relevance score, price and recommended quantity.  
*Import Requirements: preprocessor, matplotlib*

6. **evaluation.py**  
Contains functions for calculating metrics such as mean average precision (mAP), average precision (AP), precision, and recall.  
*Import Requirements: numpy*

## Usage
1. Select a recipe and process quantity information. We select a random recipe from the 1M+ Recipe dataset. 
```
i = random.randint(0, 1000000)
recipe_id, recipe = recipe_load_index(i, recipes_1M)
recipe = qty_prepocess(recipe)
```
2. Preprocess the product text associated with all products in the database as well as the recipe ingredients.
```
preprocess_df(product_df)	
preprocess(recipe['ingredient'])
```
3. Instantiate the TransformerRanker class with a base model, product embeddings, and max_rank, which is the number of products to be recommended per recipe ingredient.
```
tr = TransformerRanker(model=base_model, max_rank=k)
tr.load_embeddings(base_model_product_embeddings)
```
Note: The code for encoding the products and generating embeddings is given in the generate_embeddings.ipynb file.

4. Instantiate the Mapper and DisplayProduct class.
```
pm = Mapper(product_df)
dp = DisplayProducts(ranker=tr, mapper=pm)
```
5. Rank products using the TransformerRanker.
```
ranked_products = tr.rank_products_recipe(recipe['ingredient'].values)
```
6. Get recommended quantities for the ranked products.
```
final_recommendations = qty.recommended_quantity(ranked_list)
```
7. Display recommended products.
```
dp.display_products_df(final_recommendations)
``` 


