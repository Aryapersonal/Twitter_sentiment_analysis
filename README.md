# TASK 04 - PRODIGY DATA SCIENCE INTERNSIP PROJECT

###  SENTIMENT ANALYSIS USING TWITTER DATASET

### AUTHOR : ARYA S

##### Dataset : Twitter  dataset

##### Language: Python,Google Colab

#### Libraries:

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **matplotlib**: Data visualization.
- **seaborn**: Statistical data visualization.
- **scikit-learn**: Machine learning library.
- **nltk**: Natural language processing.
- **wordcloud**: Visualization of the most frequent words.

### **Overview**
================

This project performs sentiment analysis on a Twitter dataset using a Naive Bayes classifier. The goal is to classify tweets into four categories: Irrelevant, Negative, Neutral, and Positive. The dataset is preprocessed to remove unwanted characters and stopwords before being used to train the model. The project includes data visualization to understand the distribution of sentiments and the most frequent words in each sentiment category.

**Table of Contents**

1. [Overview](#overview)
2. [Installation](#installation)
3. [Features](#features)
4. [Project Steps](#project-steps)
5. [Visualizations and Insights](#visualizations-insights)
6. [Key Insights](#key-insights)
7. [Conclusion](#conclusion)
8. [Acknowledgments](#acknowledgments)
9. [Author Information](#Author-Information)


## Installation
To run this project,  you will need google account connected to google colab installed on your system.


### Dataset Features

- **text**: The content of the tweet.
- **sentiment**: The sentiment label assigned to the tweet (Irrelevant, Negative, Neutral, Positive).


## Project Steps

1. **Setup and Install Required Libraries**:
    - Install libraries for data manipulation, visualization, and machine learning.
    - Import necessary modules.

2. **Data Loading and Preprocessing**:
    - Load the dataset using pandas.
    - Clean the text data by removing URLs, mentions, hashtags, numbers, punctuation, and stopwords.

3. **Text Vectorization**:
    - Convert text data into numerical format using `CountVectorizer`.

4. **Model Training**:
    - Split the dataset into training and testing sets.
    - Train a Naive Bayes classifier using `MultinomialNB`.

5. **Model Evaluation**:
    - Evaluate the model using accuracy score, classification report, and confusion matrix.
    - Visualize the results using Seaborn and Matplotlib.

6. **Data Visualization**:
    - Plot the distribution of sentiments.
    - Generate word clouds for each sentiment class.
  
## Visualizations and Insights:

### Sentiment Distribution in two ways


- Frist visualization

![download](https://github.com/user-attachments/assets/e4b28e01-2178-41c8-a307-32e2d9cec261)


- Second Visualization

![download](https://github.com/user-attachments/assets/6e78fd01-da31-4b86-84a9-9d77e3bca923)



- Positive and Neutral sentiments have the highest counts, suggesting these sentiments are more prevalent in the dataset.
- Negative sentiment also has a significant count, though slightly less than Positive and Neutral.
- Irrelevant sentiment has the lowest count, indicating fewer tweets fall into this category.


### Word Cloud for each sentiment

-  Word Cloud for neutral sentiment
  
![download](https://github.com/user-attachments/assets/160b041c-0650-41b7-b9aa-b9cb172e599a)

- Word Cloud for negative sentiment

![download](https://github.com/user-attachments/assets/e205a263-235e-4b51-82c7-5501cda52518)

- Word Cloud for positive sentiment

![download](https://github.com/user-attachments/assets/74645eaa-0015-4c28-86e6-4172dcf19643)

- Word Cloud for irrelavant sentiment


![download](https://github.com/user-attachments/assets/7dc6f15f-911b-4950-9f13-c19d90fc7b93)


### Confusion Matrix

![download](https://github.com/user-attachments/assets/6bca988f-755d-4d1d-948b-529e2359c85b)


- The confusion matrix revealed that the model often misclassified tweets between the Neutral and Irrelevant classes.
- Correct predictions were highlighted on the diagonal of the matrix, while misclassifications were represented off-diagonal.


## Key Insights

1. **Model Performance**:

```
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
```

```
Accuracy: 0.45

            precision    recall  f1-score   support

  Irrelevant       0.43      0.27      0.33        37
    Negative       0.48      0.62      0.54        52
     Neutral       0.53      0.29      0.37        59
    Positive       0.39      0.60      0.47        52

    accuracy                           0.45       200
   macro avg       0.46      0.44      0.43       200
weighted avg       0.47      0.45      0.44       200
```
   
   - Naive Bayes classifier achieved 45% accuracy.
   - **Negative** tweets had the highest F1-score of 0.54.
   - **Irrelevant** and **Neutral** tweets were more challenging with F1-scores of 0.33 and 0.37.
   - **Positive** tweets had an F1-score of 0.47.
   
2. **Confusion Matrix**:
   - Revealed frequent misclassifications between **Neutral** and **Irrelevant** tweets.
   - Diagonal elements showed correct predictions, while off-diagonal elements indicated misclassifications.

3. **Sentiment Distribution**:
   - The dataset was balanced with representation across all sentiment classes.
   - The distribution helped evaluate the model’s performance and class imbalances.

4. **Word Clouds**:
   - Visualized common words for each sentiment class:
     - **Irrelevant** tweets had less sentiment-specific words.
     - **Negative** tweets contained words of dissatisfaction.
     - **Neutral** tweets were factual.
     - **Positive** tweets included words of satisfaction.


## Conclusion

The sentiment analysis project on the Twitter dataset yielded important insights and identified several valuable insights for improvement. 

## Acknowledgments
Thanks to the contributors of the libraries used in this project: Pandas, NumPy, Matplotlib, and Seaborn.

Thanks to the creators of the Twitter  dataset for providing the data used in this analysis.

Special thanks to the Prodigy Infotech to provide me this opportunity to showcase my skills in Data loading & preprocessing ,train_test_split data ,sentiment analysis Train models like classification report , accuracy score , Visualize Confusion matrix &  Sentiment distribution and forming meaningul insights.

### Author : ARYA S
### LinkedIn :  www.linkedin.com/in/arya-dataanalyst



