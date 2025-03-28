########## 1. Import required libraries ##########
import multiprocessing
import re

import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gdown
import os
nltk.download('punkt')
from nltk import word_tokenize

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy import stats


nltk.download('stopwords')
from nltk.corpus import stopwords


########## 2. Define text preprocessing methods ##########


def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Stopwords
NLTK_stop_words_list = stopwords.words('english')  # List of stopwords from NLTK library
custom_stop_words_list = ['...']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list


def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])


def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


########## 3. Download & read data ##########


# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'combined_dataset'  # Using "combined_dataset" for experiment but any file can be chosen.
path = f'./datasets/{project}.csv'

pd_all = pd.read_csv(path)  # Read csv file into dataframe
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffles data, fixed state to remove randomness

# Merge Title and Body into a single column; if Body is NaN, use Title only
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])


########## 4. Configure parameters & Start training ##########


# ========== Key Configurations ==========

# 1) Data file to read
datafile = 'Title+Body.csv'

# 2) Number of repeated experiments
REPEAT = 20

# 3) Output CSV file name
out_csv_name = f'../{project}_LR+W2V.csv'

# ========== Read and clean data ==========
data = pd.read_csv(datafile).fillna('')
text_col = 'text'

# Keep a copy for referencing original data if needed
original_data = data.copy()

# Text cleaning
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)


# Lists to store metrics across repeated runs
accuracies = []
precisions = []
recalls = []
f1_scores = []
auc_values = []

# Tokenise each every report using punkt tokeniser model from NLTK
data_tokenised = [word_tokenize(bug_report_description) for bug_report_description in data[text_col]]


# Manually train word2vecmodel
def trained_model_word2vec_embeddings(training_data):
    vector_size = 300  # The dimensions of the word embeddings
    w2v_model = gensim.models.Word2Vec(training_data,
                                       workers=1,  # can change to workers=multiprocessing.cpu_count()-1 but introduces randomness
                                       sg=1,  # Training method
                                       vector_size=vector_size,
                                       window=15,  # window size
                                       min_count=12,  # Minimum number of occurences of a word
                                       seed=999)  # Fixed state to remove randomness
    words = set(w2v_model.wv.index_to_key)

    data_vectorised = []
    # Vectorise every bug report using manually trained model
    for bug_report_description_tokenised in data_tokenised:
        bug_report_description_vectorised = np.array([w2v_model.wv[word_token]
                                                      for word_token in bug_report_description_tokenised
                                                      if word_token in words])
        if bug_report_description_vectorised.size > 0:
            # Calculate mean embedding of all words in a bug report
            data_vectorised.append(bug_report_description_vectorised.mean(axis=0))
        else:
            # If there are no words that can be vectorised in the bug report, append 0s instead
            data_vectorised.append(np.zeros(vector_size, dtype=float))
    return np.array(data_vectorised)


def google_news_model_embeddings():
    # Uses embeddings from pre-trained W2VModel trained on Google News Data
    # Downloads Google News word embeddings the first time the software is run
    url = 'https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'

    model_path = 'GoogleNews-vectors-negative300.bin.gz'
    if not os.path.exists(model_path):
        print("Model not found locally. Downloading the model...")
        gdown.download(url, model_path, quiet=False)
    else:
        print("Model already exists. Loading the model...")
    w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)
    data_vectorised = []
    # Vectorise every bug report using pre-trained model
    for bug_report_description_tokenised in data_tokenised:
        bug_report_description_vectorised = np.array([w2v_model[word_token]
                                                      for word_token in bug_report_description_tokenised
                                                      if word_token in w2v_model])
        if bug_report_description_vectorised.size > 0:
            # Calculate mean embedding of all words in a bug report
            data_vectorised.append(bug_report_description_vectorised.mean(axis=0))
        else:
            # If there are no words that can be vectorised in the bug report, append 0s instead
            data_vectorised.append(np.zeros(300, dtype=float))
    return np.array(data_vectorised)


# ========== Hyperparameter grid ==========
logistic_regression_params = {
    'solver': ['liblinear', 'lbfgs', 'newton-cg'],  # Optimisation algorithm
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse Regularisation Strength
}

word_embeddings = {"Manually Trained": trained_model_word2vec_embeddings(data_tokenised),
                   "Google News": google_news_model_embeddings()}  # Dictionary to store word embeddings for comparison

best_score = 0
best_embedding = None
for key in word_embeddings:
    grid = GridSearchCV(  # Perform grid search on logistic regression model
        LogisticRegression(class_weight='balanced', random_state=0, max_iter=500, penalty='l2'),
        logistic_regression_params,
        cv=5,  # fold 5 times
        scoring='f1'  # Performance metric to optimise for is f1
    )
    grid.fit(word_embeddings[key], data['sentiment'])
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_params = grid.best_params_  # Identifies best hyperparameters for logistic regression model
        best_embedding = key  # Identifies set of word embeddings that optimises f1

print(f"Best model: {best_embedding}")

data_vectorised = word_embeddings[best_embedding]
for repeated_time in range(REPEAT):  # 20 training and test cycles
    # --- 4.1 Split into train/test ---
    indices = np.arange(data_vectorised.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time)  # random_state parameter fixed to remove randomness

    X_train = data_vectorised[train_index, :]  # Training split for input
    X_test = data_vectorised[test_index, :]  # Test Split for input

    y_train = data['sentiment'].iloc[train_index]  # Training split for output
    y_test = data['sentiment'].iloc[test_index]  # Test split for output

    # Reassign new logistic regression model each cycle to prevent overfitting
    best_model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000, penalty='l2',
                                    **best_params)  #Uses l2 regularisation
    best_model.fit(X_train, y_train)  # Model fits training data

    y_pred = best_model.predict(X_test)  # Predict test data

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Precision (macro)
    prec = precision_score(y_test, y_pred, average='macro')
    precisions.append(prec)

    # Recall (macro)
    rec = recall_score(y_test, y_pred, average='macro')
    recalls.append(rec)

    # F1 Score (macro)
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)

    # AUC
    # If labels are 0/1 only, this works directly.
    # If labels are something else, adjust pos_label accordingly.
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    auc_values.append(auc_val)


# --- 4.5 Aggregate results ---
final_accuracy = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall = np.mean(recalls)
final_f1 = np.mean(f1_scores)
final_auc = np.mean(auc_values)


def shapiro_wilk_test(performance_scores):
    """Used to identify if results are normally distributed"""
    stat, p_value = stats.shapiro(performance_scores)
    print(f"Shapiro-Wilk Test: Stat={stat}, p-value={p_value}")

    # If p-value < 0.05, reject the null hypothesis (i.e., the data is not normal)
    if p_value < 0.05:
        print("The results are not normally distributed.")  # Null hypothesis accepted
    else:
        print("The results are normally distributed.")  # Null hypothesis rejected

# Uncomment the lines of code below to perform Shapiro-Wilk test
# shapiro_wilk_test(f1_scores)
# shapiro_wilk_test(auc_values)


print(f"=== Logistic Regression + {best_embedding} Word2Vec Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

# Save final results to CSV (append mode)
try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
        'repeated_times': [REPEAT],
        'Accuracy': [final_accuracy],
        'Precision': [final_precision],
        'Recall': [final_recall],
        'F1': [final_f1],
        'AUC': [final_auc],
        'CV_list(AUC)': [str(auc_values)]
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")

