## Functionalities:
Descriptions are provided in a brief and concise manner. For more details, please refer to the code file comments: 

1. **Ingredients_classification(nli).ipynb**  
Create hierarchical classification models for ingredients based on different levels: class, subclass, item_type. Product titles are preprocessed to remove brand and quantity information. These titles are then used to predict a selected level such as class, subclass, item_type. This is a multi-class classification task.  
*Import Requirements: preprocessor, torch, sentence_transformers, pandas, os, sys*

2. **Ingredients_generate_embeddings.ipynb**  
Generate ingredient product embeddings for the Tranformer and BM25 models which will be used in the main process code.  
*Import Requirements: preprocessor, patten_search, ranker, sentence_transformers, pandas, numpy, pickle*

3. **Gadgets_classification(nli).ipynb**  
Create hierarchical classification models for kitchen gadgets based on different levels: class, subclass, item_type. Product titles are preprocessed to remove brand and quantity information. These titles are then used to predict a selected level such as class, subclass, item_type. This is a multi-class classification task.  
*Import Requirements: preprocessor, torch, sentence_transformers, pandas, os, sys*

4. **Gadgets_generate_embeddings.ipynb**  
Generate kitchen gadeget embeddings for the Tranformer models (MiniLM, Glove, RoBERTa) which will be used in the main process code.  
*Import Requirements: preprocessor, patten_search, ranker, sentence_transformers, pandas, numpy, pickle*

5. **Gadgets_NER.ipynb**    
Creates a named entity recognition model using recipe directions for identifying the kitchen gadgets that will be used in a recipe.  
*Import Requirements: preprocessor, spacy, nltk, re, pandas, numpy*
