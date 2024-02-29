import pandas as pd
import nltk
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import pipeline
from transformers import AutoModel
from transformers import ElectraForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from simpletransformers.classification import ClassificationModel

nltk.download('stopwords')

def reading_data_set(path):
    """Reads dataset from a CSV file.

    Args:
    path (str): Path to the CSV file.

    Returns:
    DataFrame: The dataset as a Pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df

class DataProcessing:
    def __init__(self, data, column_name):
        self.df = data
        self.columns = column_name

    @staticmethod
    def clean_text(text):
        """Cleans text by removing mentions, hashtags, URLs, and other unwanted patterns.

        Args:
        text (str): The text to be cleaned.

        Returns:
        str: The cleaned text.
        """
        text = re.sub(r'@[A-Za-z0-9]+', '', str(text))
        text = re.sub(r'#', '', str(text))
        text = re.sub(r'RT[\s]+', '', str(text))
        text = re.sub(r'https?:\/\/\S+', '', str(text))
        text = re.sub(r'\n', ' ', str(text))
        text = re.sub(r'[\xa0]+', '', str(text))
        text = re.sub(r'- dunya.com', '', str(text))
        return text

    def clean_data(self, subs):
        """Preprocesses the data by cleaning text and tokenizing.

        Args:
        subs (str): The column name containing the text data.

        Returns:
        DataFrame: Processed DataFrame with cleaned text.
        """
        self.df = self.df.rename(columns={'Sub Cat': subs})
        self.df[self.columns] = self.df[self.columns].replace('\n', '').replace('/', '')
        self.df = self.df.dropna()
        processed_dataset_by_category = {}
        unique_words_by_class_with_count = {}
        regex_tokenizer = nltk.RegexpTokenizer(r"\w+")
        turkish_stop_words = stopwords.words('turkish')
        for index, row in self.df.iterrows():
            unique_words_by_class_with_count.setdefault(row[subs], {})
            processed_dataset_by_category.setdefault(row[subs], [])
            tokenized_words = regex_tokenizer.tokenize(row[subs])
            tokenized_words = [word for word in tokenized_words if word not in turkish_stop_words]
            processed_dataset_by_category[row[subs]].append(tokenized_words)
            for word in tokenized_words:
                unique_words_by_class_with_count[row[subs]].setdefault(word, 0)
                unique_words_by_class_with_count[row[subs]][word] += 1
        dataset_x = []
        dataset_y = []
        categories = list(processed_dataset_by_category.keys())
        default_label = [0 for i in range(len(categories))]
        for category, docs in processed_dataset_by_category.items():
            for doc in docs:
                dataset_x.append(" ".join(doc))
                label = default_label.copy()
                label[categories.index(category)] = 1
                dataset_y.append(label)
        df = pd.DataFrame()
        df['processed'] = dataset_x
        df['labels'] = self.df['subs']
        return df

    @staticmethod
    def label_encoding(df):
        """Encodes labels into numerical values using LabelEncoder.

        Args:
        df (DataFrame): DataFrame containing label column.

        Returns:
        DataFrame: DataFrame with encoded labels.
        """
        encoder = LabelEncoder()
        df['labels'] = encoder.fit_transform(df.labels)
        return df

    @staticmethod
    def split_dataset(df):
        """Splits dataset into train and test sets.

        Args:
        df (DataFrame): DataFrame containing processed data.

        Returns:
        DataFrame, DataFrame: Train and test sets.
        """
        train, test = train_test_split(df, test_size=0.3, random_state=0)
        return train, test

class Model:
    def __init__(self, model_name):
        self.model = model_name

    def sentiment_analysis(self, data, content_column_name, date_time):
        """Performs sentiment analysis on the given data.

        Args:
        data (DataFrame): DataFrame containing text data.
        content_column_name (str): Name of the column containing text.
        date_time (str): Name of the column containing date/time information.

        Returns:
        DataFrame: DataFrame with sentiment analysis results.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        models = AutoModelForSequenceClassification.from_pretrained(self.model)
        sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=models)
        savasy_list = []
        for i, row in tqdm(data.iterrows(), total=len(data)):
            try:
                sent_an = sa(row[content_column_name])
                norm_val = sent_an[0]['score']
                savasy_list.append([sent_an[0]['label'], norm_val, row[content_column_name], row[date_time]])
            except:
                continue
        savsy_df = pd.DataFrame(savasy_list, columns=['label', 'score', 'article', 'date_time'])
        return savsy_df

    def classification_turkish_text(self, num_of_classes, epoch_number, run_gpu, train_dataset, test_dataset):
        """Performs text classification using BERT model.

        Args:
        num_of_classes (int): Number of classes for classification.
        epoch_number (int): Number of training epochs.
        run_gpu (bool): Flag to indicate if GPU should be used.
        train_dataset (DataFrame): DataFrame containing training data.
        test_dataset (DataFrame): DataFrame containing testing data.

        Returns:
        array, array: Actual and predicted labels.
        """
        model = ClassificationModel('bert', self.model, num_labels=num_of_classes, use_cuda=run_gpu,
                                    args={'reprocess_input_data': True, 'overwrite_output_dir': True,
                                          'num_train_epochs': epoch_number, "train_batch_size": 64,
                                          "fp16": False, "output_dir": "bert_model"})
        train_model = model.train_model(train_dataset)
        result, model_outputs, wrong_predictions = model.eval_model(test_dataset)
        predictions = model_outputs.argmax(axis=1)
        actuals = test_dataset.subs.values
        return actuals, predictions
