# NLP record filtering reproduction and analysis

NLP Filtering reproduction and analysis was made in May 2021 due to the reviewer request. Analysis reproduction is made to analyze the difference between the experts' selection and classifier selection. The analysis is divided into two parts. At first, the model is retrained using a training data set. It is the same training data set with 470 records (provided in attachments). Then the model was applied to the records found during the search to classify records. The classification result was compared with the final data set, which was selected manually for analysis. A crude data set of search results is available on demand.  Lower case titles of classified papers from search results were matched to the final selection paper titles using Levenshtein distance criteria. Titles were considered equal if the distance is less than 5.   

Model training procedure was almost the same as in the initial filtering with little differences:

* Gradient Boosting was used instead of Random Forest.
* Word embedding techniques were used in the vectorization. One dimensional co-occurrence matrix was used instead of word index in vector.
* Table with final selection for full-text analysis didn't survive in its initial state. Instead of the 621-records version, a slightly filtered version with 612 records was used.
* Table of search results was gathered together from different initial files filled during the search and resulted in 9932 records with duplications.

## File descriptions

* exclude.csv - filter keywords
* full_record.csv - crude search results data set
* io_part.py - code for file saving and loading
* model_train.py - **start file**
* model_update.mdl - trained GB model 
* nlp_part_shadow.py - vectorization lib
* selected.csv - final selection data set
* studies_checks.py - lib that classfies search results using model and compare with final selection
* training_data.csv - data that were used to train model 


## Training results

Model was trained with AUC ROC = 0.93 over the test sample (30\% of the training set). 

|Threshold|Eligible|Not Eligible|TP|FP|TN |FN|TPR     |FPR     |F1         |
|---------|--------|------------|--|--|---|--|--------|--------|-----------|
|0.05     |  35    |         106|33|28| 78| 2|0.942857|0.264151|  0.6875   | 
|0.1      |35      |         106|32|20| 86| 3|0.914286|0.188679|  0.735632 |
|0.15     |  35    |         106|30|13| 93| 5|0.857143|0.122642|  0.769231 | 
|0.2      |35      |         106|30|11| 95| 5|0.857143|0.103774|  0.789474 |
|0.25     |  35    |         106|29| 9| 97| 6|0.828571|0.084906|  0.794521 | 
|0.3      |35      |         106|28| 7| 99| 7|0.8     |0.066038|  0.8      |
|0.35     |  35    |         106|28| 4| 02|7 |0.8     |0.037736|  0.835821 | 
|0.4      |35      |         106|28| 4|102|7 |0.8     |0.037736|  0.835821 |
|0.45     |  35    |         106|27| 4|102|8 |0.771429|0.037736|  0.818182 |
|0.5      |35      |         106|26| 2|104|9 |0.742857|0.018868|  0.825397 |
|0.55     |  35    |         106|24| 2|104|11|0.685714|0.018868|  0.786885 |
|0.6      |35      |         106|23| 2|104|12|0.657143|0.018868|  0.766667 |
|0.65     |  35    |         106|22| 1|105|13|0.628571|0.009434|  0.758621 |
|0.7      |35      |         106|20| 1|105|15|0.571429|0.009434|  0.714286 |
|     0.75|  35    |         106|19| 1|105|16|0.542857|0.009434|  0.690909 |
|     0.8 |35      |         106|19| 1|105|16|0.542857|0.009434|  0.690909 |
|     0.85|  35    |         106|19| 0|106|16|0.542857|0       |  0.703704 |
|     0.9 |35      |         106|14| 0|106|21|0.4     |0       |  0.571429 |
|     0.95|  35    |         106|10| 0|106|25|0.285714|  0     |  0.444444 |
|     1   |  35    |         106|0|  0|106|35|0       |  0     |  0        |

## Search data classification results


|                                                                | TPR   | FPR   |
|--------------------------------------------------------------- |-------|-------|
|Model testing                                                   | 0.8   | 0.07  |
|Search results classification comparison with manually selected | 0.747 | 0.128 |

The classifier shows worse results on the "real data." There are some reasons for that. The main reason is that the training data set was based on the Pubmed search results.  For quite many records from WoS no abstract was available (only shortened version of abstract). Moreover, specific Pubmed keywords were not available for studies that were presented only at Scopus and WoS. These differences affected classification performance.

Also, it is worth mentioning that only 566 of 612 records were matched to the final selection during the classification of search data. There are several reasons for that.  First, several unique studies were found in the relevant systematic reviews bibliographies. Thus they were not presented in the search data, but they are presented in final selection data. This reason covers 50\% of "missed" studies. Second, some titles contain specific symbols that confuse matching criteria. Titles are in the search results, but they can't be matched to the final selection titles.
