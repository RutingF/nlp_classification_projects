# nlp_sentiment_classification
Includes classification models trained with count vector or TF-IDF vector, using AUC as evaluation metric

## sentiment_classification Module

Evaluating 5 different combinations of models & features: 
* Multinomial Naive Bayes classifier model w/ Count Vectorizer
* Multinomial Naive Bayes classifier model w/ TFI-DF Vectorizer 
* SVM w/ TF-IDF Vectorizer and an additional feature (document length)
* Logistic Regression model w/ Tfidf Vectorizer 
    * Additional Features:
        * Document length
        * Number of digits per document 

* Logistic Regression model w/ Count Vectorizer
    * Additional Features:
        * Document length
        * Number of digits per document
        * Number of non-word characters 

### Measurement
AUC (Area Under the Curve)


<b> sentiment_analysis.ipynb </b>

Jupyter notebook demonstrating the usages of <b> sentiment_classification.py </b>

