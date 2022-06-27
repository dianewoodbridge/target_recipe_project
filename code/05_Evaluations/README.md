Descriptions are provided in a brief and concise manner. For more details, please refer to the code file comments: 

1. **01_search_evaluation.ipynb**  
Evaluates different Rankers using mean average precision (MAP@K) based on the results for the top ingredients across the Recipe1M+ corpus.  
*Import Requirements: preprocessor, evaluation, ranker, sentence_transformers, mapper, display_products*

2. **02_Qty_rec_evaluation.ipynb**  
Process to evaluate quantity recommendations for multiple top ingredients.  
*Import Requirements: preprocessor, recipe_loading, qty_mapping, ranker, sentence_transformers, mapper, display_products*

3. **03_MAP_For_recipes.ipynb**  
Retrieves 100 recipes that contain the top 1000 ingredients in the Recipe1M+ corpus.   
*Import Requirements: preprocessor, recipe_loading, qty_mapping, ranker, sentence_transformers, mapper, display_products*

4. **04_search_evaluation-kitchen_gadgets.ipynb**  
Evaluates different Rankers using mean average precision (MAP@K) based on the results for the top kitchen gadgets across the Recipe1M+ corpus.  
*Import Requirements: preprocessor, evaluation, ranker, sentence_transformers, mapper, display_products*
