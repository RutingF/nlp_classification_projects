# nlp_classification_projects

This repository demonstrates how to leverage classification models in Natural Language Processing (NLP) for two prevalent Machine Learning (ML) applications: 
* Sentiment Classificaiton
* Spam Detection 

## Classification Module Overview 

The 'TextClassification' class offers a wide range of functionalities for text preprocoessing, feature engineering, model training, and evaluation. 

### Key Features 
- **Data Preprocessing**: Handle missing values, encode ratings for sentiment analysis, and preprocess text data for classification.
- **Feature Engineering**: Extract additional features such as document length, number of digits, and non-word characters.
- **Vectorization**: Utilize Count Vectorizer and TF-IDF Vectorizer to convert text data into numerical features.
- **Classification Models**: Train various classification models including Logistic Regression, Multinomial Naive Bayes, and Support Vector Machine (SVM).
- **Evaluation Metrics**: Assess model performance using Area Under the Curve (AUC) metric.

### Sentiment Classification 
Evaluation involves 2 different cobinations of models & features: 

* Logistic Regression model with Count Vectorizer 
* Logistic Regression model with TF-IDF Vectorizer 

### Spam Detection 
Evaluation involves 5 different combinations of models & features: 
* Multinomial Naive Bayes classifier model with Count Vectorizer
* Multinomial Naive Bayes classifier model with TF-IDF Vectorizer 
* SVM with TF-IDF Vectorizer and an additional feature (document length)
* Logistic Regression model with TF-IDF Vectorizer 
    * Additional Features:
        * Document length
        * Number of digits per document 

* Logistic Regression model with Count Vectorizer
    * Additional Features:
        * Document length
        * Number of digits per document
        * Number of non-word characters 



## Sample Data Sets 
- **Sentiment analysis**: User review data, where any rating above 3 is encoded as "positively rated". This is a snippet sample of the original data used for model training. 
- **Spam detection**: Raw text with 'spam' / 'nonspam' labels. 


## classification_demo.ipynb

Jupyter notebook demonstrating the usage of **classification.py**. 



