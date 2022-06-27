Descriptions are provided in a brief and concise manner. For more details, please refer to the code file comments. The API has the following classes and functionalities that are implemented in the associated python files: 

1. **recipe_loading.py**
<br>Loads the recipe json file and tabularises the ingredients, required quantity, and the kitchen gadgets to be sued as an input in the further processes.</br>
*Import Requirements: requests, json, re, pandas, numpy*

2. **pattern_search.py**  
Contains the PatternMatcher class and associated functions for pattern matching. The functions associated include searching for products using exact string matching, stemmed string matching, searching for hypernyms, hyponyms, and food nouns.  
*Import Requirements: re, nltk, spacy*

3. **preprocessor.py**  
Contains functions for preprocessing ingredient and product query text. Some of the preprocessing functionalities that could be applied include lowercasing, stemming, tokenization, removing punctuations, stop words, quantities, brand information. There is also a query expansion functionality that can expand an ingredient query with the hypernyms, hyopnyms, synonyms, and food nouns that are identified by the WordNet lexical database.  
*Import Requirements: pattern_search, pandas, numpy, string, ast, sklearn*

4. **ranker.py**    
Contains the classes Ranker, TransformerRanker, CrossEncoderRanker, BM25Ranker, Classifier, RankerCombination, and RankerPipeline.
    - Ranker: Base Ranker class with common functionalities for TransformerRanker, CrossEncoderRanker and BM25Ranker classes. 
    - TransformerRanker:  This is the level 1 transformer ranker for quick candidate retrieval
    - CrossEncoderRanker: This is the level 2 transformer ranker for re-ranking. It takes in the level 1 transformer ranker as an argument for quick candidate retrieval after which it re-ranks using a cross-encoder. The cross-encoder gives a logit score for a pair of texts indicating their similairity.
    - BM25Ranker: This ranker can be used as a level 1 ranker for quick candidate retrieval. 
    - Classifier: This is the hierarchical classification class that contains functionalities for filtering products based on the product category that an ingredient most likely belongs to. 
    - RankerPipeline: Provides functionality to use different rankers (BM25Ranker, TransformerRanker, CrossEncoderRanker) in a sequential stagewise manner.
    - RankerCombination: Provides functionality to combine rankers using an array of weights corresponding to the rankers.    

*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Import Requirements: preprocessor, rank_bm25, sentence_transformers, torch, joblib*

5. **mapper.py**    
Contains the Mapper class that contains functions for mapping from a product_id to an image, title, or any other dataframe column. 
*Import Requirements: PIL, requests, io, numpy*

6. **display_products.py**  
Contains the DisplayProducts class that can display the most relevant products for a recipe or ingredient based on a ranker. It can also display products based on a product_id or list of product_ids. The products are displayed along with the relevance score, price and recommended quantity.  
*Import Requirements: preprocessor, matplotlib*

7. **qty_mapping.py**
<br>Standardises the units (cup/tablespoon/teaspoon) used in recipe into a standard volume scale. Loads density information to calculate weight required from the volume. Finally, calculate the required amount of each ingredient. </br>
*Import Requirements: re, pandas, numpy*

8. **evaluation.py**  
Contains functions for calculating metrics such as mean average precision (mAP), average precision (AP), precision, and recall.  
*Import Requirements: numpy*
