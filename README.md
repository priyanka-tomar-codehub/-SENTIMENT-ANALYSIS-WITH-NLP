# -SENTIMENT-ANALYSIS-WITH-NLP
COMPANY:CODTECH IT SOLUTIONS NAME:PRIYANKA TOMAR INTERN ID:CT04DY481 DOMAIN:MACHINE LEARNING DURATION:4 WEEKS MENTOR:NEELA SANTOSH
##Description of the Task: Sentiment Analysis on Customer Reviews using Logistic Regression

The task focuses on building a sentiment analysis model that classifies customer reviews as either positive or negative using Natural Language Processing (NLP) techniques and a Logistic Regression classifier. This project demonstrates the process of converting unstructured text data into meaningful numerical representations, training a machine learning model, and evaluating its performance. The entire workflow includes data exploration, preprocessing, feature extraction, model training, and evaluation.

Dataset Overview

The dataset, named customer_reviews.csv, consists of two main columns:

review: Textual customer feedback provided in natural language.

sentiment: The corresponding label, indicating whether the review is positive or negative.

Understanding customer opinions is crucial for businesses because it helps assess product quality, customer satisfaction, and areas of improvement. Automating this task with sentiment analysis provides quick, scalable, and efficient insights from large volumes of textual data.

Data Exploration and Preprocessing

After loading the dataset into a Pandas DataFrame, the first step involved checking the structure of the data using .head() and examining its size with .shape. The distribution of sentiments was then analyzed to understand whether the dataset is balanced or imbalanced. A count plot visualization was generated to show the number of positive and negative reviews. Balanced datasets ensure that the model does not become biased toward one class, while imbalanced datasets may require resampling techniques.

Feature Extraction using TF-IDF

Since machine learning algorithms cannot directly interpret text data, the reviews were transformed into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency). TF-IDF is a statistical method that reflects how important a word is to a document relative to a collection of documents.

Term Frequency (TF): Measures how often a word appears in a review.

Inverse Document Frequency (IDF): Reduces the importance of words that appear frequently across all reviews (e.g., "good", "very").

By applying TfidfVectorizer, each review was represented as a sparse matrix of weighted word frequencies. To improve efficiency and prevent overfitting, the feature space was limited to the top 5,000 words. Common English stopwords (like the, is, and, in) were removed automatically to ensure only meaningful words contributed to classification.

Model Training using Logistic Regression

A Logistic Regression classifier was selected for this task. Logistic Regression is a simple yet powerful algorithm for binary classification problems. It models the probability that a given input belongs to one of two categories by applying the logistic (sigmoid) function. In this project, it was used to predict whether a review expressed a positive or negative sentiment.

The training dataset (80% of the total data) was used to fit the model, while the test dataset (20%) was reserved for evaluation. The model was trained with maximum iterations set to 1000 to ensure convergence given the high-dimensional TF-IDF feature space.

Model Evaluation

The trained model was tested on unseen data to measure its performance. The following evaluation metrics were calculated:

Accuracy: The percentage of correctly classified reviews.

Classification Report: Detailed metrics such as precision, recall, and F1-score for both positive and negative sentiments. Precision indicates the proportion of correctly predicted positive reviews, recall measures the proportion of actual positive reviews correctly identified, and the F1-score balances precision and recall.

Additionally, a confusion matrix was computed and visualized as a heatmap. The confusion matrix provides a clear breakdown of true positives, true negatives, false positives, and false negatives, making it easier to analyze misclassifications.

Results and Conclusion

The model achieved a strong accuracy score, demonstrating that Logistic Regression with TF-IDF is effective for sentiment analysis. Positive and negative reviews were classified with high precision and recall, though some misclassifications occurred due to ambiguous or sarcastic language in the reviews.

This task highlights how machine learning and NLP techniques can convert raw textual data into actionable insights. By automating sentiment classification, businesses can analyze customer opinions at scale, leading to improved decision-making and customer experience management. Future improvements could include experimenting with more advanced models such as Support Vector Machines (SVM), Random Forests, or even deep learning approaches like Recurrent Neural Networks (RNNs) and Transformers for capturing more complex linguistic patterns.

##output
