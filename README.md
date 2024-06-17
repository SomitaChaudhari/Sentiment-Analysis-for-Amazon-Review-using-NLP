## Sentiment Analysis for Amazon Reviews
[![DOI](https://zenodo.org/badge/814169800.svg)](https://zenodo.org/doi/10.5281/zenodo.11987919) | ***Natural Language Processing***


**Project Description**: In the rapidly evolving landscape of e-commerce, businesses face the challenge of 
harnessing the wealth of customer sentiments embedded within online reviews to drive informed decision-
making. The proliferation of product reviews on platforms like Amazon presents a double-edged sword: 
while these reviews offer invaluable insights into consumer preferences, the sheer volume and diversity of 
data make it difficult for businesses to extract meaningful and actionable insights efficiently.

**Concepts**: Natural Language Processing (NLP),
Text Preprocessing,
Sentiment Analysis,
Word Cloud Visualization,
Time Series Analysis,
Supervised Machine Learning,
Classification.

**Libraries Used**: NumPy, 
Pandas,
Matplotlib,
Seaborn,
PIL,
NLTK,
Textblob,
Wordcloud,
re,
spaCy,
tabulate,
scikit-learn,
imbalanced-learn.

**Framework**: Google Collab

**Methodology** : 
- Data Loading and Exploration
- Text Preprocessing 
   - Cleaning (removing punctuation, digits, URLs, etc.)
   - Tokenization
   - Lemmatization
   - Expanding Contractions
   - Removing Stopwords
- Exploratory Data Analysis (EDA)
   - Visualizing Review Trends Over Time
   - Word Frequency Analysis
   - Word Cloud Generation
- Feature Engineering
  - TF-IDF Vectorization
  - Sentiment Analysis
- Data Balancing
  - Random Undersampling
  - SMOTE Oversampling
- Model Building and Evaluation
  - Logistic Regression
  - Linear SVM
  - Decision Tree
  - Random Forest
  - Cross-Validation
  - Confusion Matrix
  - Classification Report

**Steps**:
1. Load the dataset and perform initial data exploration.
2. Preprocess the text data by cleaning, tokenizing, lemmatizing, and removing stopwords.
3. Conduct EDA by visualizing review trends over time, word frequencies, and generating word clouds.
4. Extract features from the text data using TF-IDF vectorization and sentiment analysis.
5. Balance the data using random undersampling and SMOTE oversampling techniques.
6. Build and evaluate various machine learning models (Logistic Regression, Linear SVM, Decision Tree, Random Forest) for sentiment classification.
7. Perform cross-validation to assess model performance and generate classification reports and confusion matrices.
8. Analyze the results and compare the performance of different models.

**Results**:
1. Without Cross Validation:
 - Logistic Regression: The accuracy of the logistic regression model is about 70%. It shows balanced performance with constant precision, recall, and F1-score in all three sentiment classes. In contrast to good sentiments, it has a little trouble categorising negative sentiments.
- Linear SVM: With an accuracy of roughly 71%, the linear support vector machine (SVM) model  performs better than logistic regression. It performs better in terms of recall and precision when classifying both positive and negative emotions, although it excels most at classifying pleasant emotions. This demonstrates how well the model can divide sentiment classes inside the feature space.
-  Decision Tree: The accuracy of the decision tree model is approximately 68%. Its accuracy and recall for neutral thoughts are reasonable, but when it comes to negative sentiments, it performs poorly overall. The models inferior performance could be attributed to overfitting, as evidenced by the differences in recall and precision scores across sentiment classes.
- Random Forest: The accuracy of the random forest model is about 69%. Robust performance is  demonstrated by the balanced precision, recall, and F1-score across emotion classes. But just like the decision tree model, it has trouble categorizing negative emotions, which lowers the precision 17 score for this class. Despite this, compared to individual decision trees, the random forest model performs better overall due to its ensemble aspect, which helps reduce overfitting.

2. With Cross Validation:
- Logistic Regression: The logistic regression model performs well in sentiment classification, with an accuracy of 0.70 and a mean cross-validation score of about 0.70. It consistently demonstrates a capacity to categories sentiments by maintaining balanced precision, recall, and F1-scores across sentiment classes.
- The linear SVM model performs well in sentiment classification, with a mean cross-validation score of around 0.71 and an accuracy of 0.71. It exhibits great precision, recall, and F1-scores across sentiment classes, particularly for positive thoughts. This indicates that sentiment classes can be effectively separated in the feature space, as well as good generalization to previously encountered data.
- Despite having a mean cross-validation score of around 0.70, the decision tree model's accuracy is significantly lower at 0.63, indicating moderate performance in sentiment categorization. It struggles with negative emotions, resulting in lower precision, recall, and F1-scores in this class. Furthermore, its accuracy is lower than other models, indicating a restricted ability to identify sentiment complexity.
- Random Forest: With a mean cross-validation score of around 0.73 and an accuracy of 0.74, the random forest model does well in sentiment classification. It maintains a balanced precision, recall, and F1-score across sentiment classes, showing strong and consistent results.
