# nlp_classification

This repository demonstrates how to leverage classification models in Natural Language Processing (NLP) for two prevalent Machine Learning (ML) applications: 
* Sentiment Classificaiton
* Spam Detection 


## Sample Data Sets 
- Sentiment analysis: User review data, where we encode any rating above 3 to be "positively rated". 
- Spam detection: raw text with 'spam' / 'nonspam' labels. 

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

