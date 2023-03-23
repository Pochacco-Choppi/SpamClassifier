
# SpamClassifier

This project is a simple implementation of text classification using Python, Flask and Pandas. It allows the user to classify text as either SPAM or NOT SPAM. The project uses NLP techniques and machine learning algorithms to analyze the text and make the classification.

# Data Preparation
The project uses a dataset of emails to train the classification model. The dataset is stored in a CSV file called `emails.csv`. The file contains two columns: `email` and `label`. The `email` column contains the email text, and the `label` column contains the classification label (either SPAM or NOT SPAM).
The data preparation module uses Pandas and Numpy to preprocess the data. It performs the following tasks:

1. Load the data from the CSV file.
2. Clean the text by removing stop words, punctuation, and numbers.
3. Vectorize the text using the Bag of Words technique.
4. Split the data into training and testing sets.
5. Train a classification model using the training data.
6. Evaluate the model using the testing data.
7. Save the trained model to a file for later use.

# Models scores
<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>model</th>      <th>accuracy</th>      <th>precision</th>      <th>recall</th>      <th>F1-score</th>    </tr>  </thead>  <tbody>    <tr>      <th>3</th>      <td>MultinomialNB1</td>      <td>0.963</td>      <td>0.961</td>      <td>0.897</td>      <td>0.928</td>    </tr>    <tr>      <th>5</th>      <td>MultinomialNB3 - tokenizer</td>      <td>0.962</td>      <td>0.959</td>      <td>0.896</td>      <td>0.926</td>    </tr>    <tr>      <th>7</th>      <td>SGDClassifier1 + chi</td>      <td>0.954</td>      <td>0.903</td>      <td>0.937</td>      <td>0.92</td>    </tr>    <tr>      <th>8</th>      <td>SGDClassifier2 + tfidf</td>      <td>0.956</td>      <td>0.957</td>      <td>0.879</td>      <td>0.916</td>    </tr>    <tr>      <th>0</th>      <td>RandomForestClassifier1 + tfidf</td>      <td>0.952</td>      <td>0.94</td>      <td>0.88</td>      <td>0.909</td>    </tr>    <tr>      <th>2</th>      <td>RandomForestClassifier2 + chi</td>      <td>0.944</td>      <td>0.907</td>      <td>0.894</td>      <td>0.9</td>    </tr>    <tr>      <th>1</th>      <td>RandomForestClassifier1 + tfidf - tokenizer</td>      <td>0.938</td>      <td>0.893</td>      <td>0.896</td>      <td>0.894</td>    </tr>    <tr>      <th>6</th>      <td>MultinomialNB4 + tfidf - tokenizer</td>      <td>0.888</td>      <td>0.997</td>      <td>0.569</td>      <td>0.725</td>    </tr>    <tr>      <th>4</th>      <td>MultinomialNB2 + tfidf</td>      <td>0.881</td>      <td>0.997</td>      <td>0.543</td>      <td>0.703</td>    </tr>  </tbody></table>
Since we need to minimize NON-SPAM emails getting into the SPAM folder, Precision is an important parameter when evaluating models. After comparing all the models, the most successful in our case was MultinomialNB1.

*test size = 20%

## Installation

To install QuickLink, follow these steps:

1. Clone the SpamClassifier repository from GitHub.
2. Install the required Python packages using pip: `pip install -r requirements.txt`
3. Run fit process: `python src/email_classifier.py`
4. Run web server for classification: `python src/classifier.py`
5. Access SpamClassifier in your web browser at http://localhost:8000.

## Run tests
You can run test using these commands:
```bash
python -m unittest tests/test_np.py
python -m unittest tests/test_pd.py
```