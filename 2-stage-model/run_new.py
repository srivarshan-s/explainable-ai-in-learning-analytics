import re
import pandas as pd
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow import keras


def remove_non_arguments(df):
    statements = df[df["argumentative"] ==
                    "n"]["argument"].to_numpy()  # Extract the statements
    statements = np.unique(statements)  # Extract the unique statements
    for ele in statements:  # Remove all occurrences of NOT argumentative statements
        df.drop(df[df['argument'] == ele].index, axis=0, inplace=True)
    return df


def combine_annotator_scores(df):
    argument = np.unique(df["argument"])
    attributes = ["annotator", "overall quality", "cogency",
                  "effectiveness", "reasonableness", "argument", "#id"]
    cleaned_df = []
    for arg in argument:
        new_df = df[df["argument"] == arg][attributes]
        flag = 0
        new_dict = {
            "#id": new_df["#id"].iloc[0],
            "argument": new_df["argument"].iloc[0],
        }
        for ele in ["overall quality", "cogency", "effectiveness", "reasonableness"]:
            if len(pd.value_counts(new_df[ele])) == 3:
                flag = 1
                break
            new_dict[ele] = pd.value_counts(new_df[ele]).index[0]
        if flag == 1:
            continue
        cleaned_df.append(new_dict)
    cleaned_df = pd.DataFrame(cleaned_df)
    return cleaned_df


def import_data():
    df = pd.read_csv("data/dagstuhl-15512-argquality-corpus-annotated.csv",
                     sep='\t', encoding_errors="ignore")
    df = remove_non_arguments(df)
    cleaned_df = combine_annotator_scores(df)
    return cleaned_df


def clean_text(text):
    text = text.replace('</br>', '')  # Remove </br>
    text = re.sub(r'[^\w]', ' ', text)  # Remove symbols
    text = re.sub(r'[ ]{2,}', ' ', text)  # Remove extra spaces
    text = re.sub(r'[ \t]+$', '', text)  # Remove trailing white spaces
    tokens = []
    stop_words = set(stopwords.words("english"))
    english_stemmer = SnowballStemmer("english")
    for token in text.split():
        if token not in stop_words:
            token = english_stemmer.stem(token)
            tokens.append(token)
    return " ".join(tokens)


def preprocess_data(cleaned_df):
    text = cleaned_df["argument"]
    cleaned_text = [clean_text(text) for text in text]
    text = cleaned_text
    return text


def vectorize_text_data(text):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    X = X.toarray()
    return X


class CustomModel:
    def __init__(self, attr_1_model, attr_2_model, attr_3_model, arg_model):
        self.attr_1_model = attr_1_model
        self.attr_2_model = attr_2_model
        self.attr_3_model = attr_3_model
        self.arg_model = arg_model

    def predict(self, array):
        attr_1 = self.attr_1_model.predict(array, verbose=0)
        attr_2 = self.attr_2_model.predict(array, verbose=0)
        attr_3 = self.attr_3_model.predict(array, verbose=0)
        attr_1 = self.__decode(attr_1)
        attr_2 = self.__decode(attr_2)
        attr_3 = self.__decode(attr_3)
        array = self.__transform(attr_1, attr_2, attr_3, array)
        pred = self.arg_model.predict(array)
        return pred

    def __decode(self, array):
        new_array = []
        label_map = {
            0: "1 (Low)",
            1: "2 (Average)",
            2: "3 (High)",
        }
        for ele in array:
            new_array.append(label_map[np.argmax(ele)])
        return np.array(new_array)

    def __transform(self, attr_1, attr_2, attr_3, array):
        attr_1 = self.__encode(attr_1)
        attr_2 = self.__encode(attr_2)
        attr_3 = self.__encode(attr_3)
        array_new = []
        for idx, ele in enumerate(array):
            temp = np.concatenate((attr_1[idx], attr_2[idx], attr_3[idx], ele))
            array_new.append(temp)
        array = np.array(array_new)
        return array

    def __encode(self, array):
        new_array = []
        label_map = {
            "1 (Low)": np.array([0, 0, 1]),
            "2 (Average)": np.array([0, 1, 0]),
            "3 (High)": np.array([1, 0, 0]),
        }
        for ele in array:
            new_array.append(label_map[ele])
        return np.array(new_array)


def encode(array, attr_enc_map):
    temp_list = []
    for ele in array:
        temp_list.append(attr_enc_map[ele])
    return np.array(temp_list)


def cross_validate(X, cleaned_df):
    kf = KFold(n_splits=5)
    pred = []
    attr_enc_map = {
        "1 (Low)": np.array([0, 0, 1]),
        "2 (Average)": np.array([0, 1, 0]),
        "3 (High)": np.array([1, 0, 0]),
    }
    y_cog = cleaned_df["cogency"].to_numpy()
    y_eff = cleaned_df["effectiveness"].to_numpy()
    y_reas = cleaned_df["reasonableness"].to_numpy()
    y_oq = cleaned_df["overall quality"].to_numpy()
    encoder = OneHotEncoder()
    enc_y_cog = encoder.fit_transform(y_cog.reshape(-1, 1)).toarray()
    enc_y_eff = encoder.fit_transform(y_eff.reshape(-1, 1)).toarray()
    enc_y_reas = encoder.fit_transform(y_reas.reshape(-1, 1)).toarray()
    encoder = LabelEncoder()
    enc_y_oq = encoder.fit_transform(y_oq)
    cogency = encode(y_cog, attr_enc_map)
    effectiveness = encode(y_eff, attr_enc_map)
    reasonableness = encode(y_reas, attr_enc_map)
    X_new = []
    for idx, x in enumerate(X):
        temp = np.concatenate(
            (cogency[idx], effectiveness[idx], reasonableness[idx], x))
        X_new.append(temp)
    X_new = np.array(X_new)
    for train_index, test_index in kf.split(X):
        # NN TO PREDICT COG
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = enc_y_cog[train_index], enc_y_cog[test_index]
        cog_model = keras.models.Sequential([
            keras.layers.Dense(
                32, input_dim=X_train.shape[1], activation="relu"),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(3, activation="softmax"),
        ])
        loss_function = keras.losses.CategoricalCrossentropy()  # Define loss function
        optimizer = keras.optimizers.SGD(
            learning_rate=0.005)  # Define optimizer
        cog_model.compile(optimizer=optimizer, loss=loss_function, metrics=[
                          "accuracy"])  # Compile the model
        cog_model.fit(X_train, y_train, epochs=75, batch_size=1,
                      validation_data=(X_test, y_test), verbose=0)
        # NN TO PREDICT EFF
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = enc_y_eff[train_index], enc_y_eff[test_index]
        eff_model = keras.models.Sequential([
            keras.layers.Dense(
                32, input_dim=X_train.shape[1], activation="relu"),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(3, activation="softmax"),
        ])
        loss_function = keras.losses.CategoricalCrossentropy()  # Define loss function
        optimizer = keras.optimizers.SGD(
            learning_rate=0.005)  # Define optimizer
        eff_model.compile(optimizer=optimizer, loss=loss_function, metrics=[
                          "accuracy"])  # Compile the model
        eff_model.fit(X_train, y_train, epochs=75, batch_size=1,
                      validation_data=(X_test, y_test), verbose=0)
        # NN TO PREDICT REAS
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = enc_y_reas[train_index], enc_y_reas[test_index]
        reas_model = keras.models.Sequential([
            keras.layers.Dense(
                32, input_dim=X_train.shape[1], activation="relu"),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(3, activation="softmax"),
        ])
        loss_function = keras.losses.CategoricalCrossentropy()  # Define loss function
        optimizer = keras.optimizers.SGD(
            learning_rate=0.005)  # Define optimizer
        reas_model.compile(optimizer=optimizer, loss=loss_function, metrics=[
                           "accuracy"])  # Compile the model
        reas_model.fit(X_train, y_train, epochs=75, batch_size=1,
                       validation_data=(X_test, y_test), verbose=0)
        # LR TO PREDICT OQ
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = enc_y_oq[train_index], enc_y_oq[test_index]
        oq_model = LogisticRegression(
            C=0.1, dual=False, fit_intercept=True, penalty="l2", solver="newton-cg")
        oq_model.fit(X_train, y_train)
        # 2-LAYER MODEL
        encoder = LabelEncoder()
        enc_y_oq = encoder.fit_transform(y_oq)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = enc_y_oq[train_index], enc_y_oq[test_index]
        custom_model = CustomModel(cog_model, eff_model, reas_model, oq_model)
        pred_test = custom_model.predict(X_test).tolist()
        pred += pred_test
    print(classification_report(enc_y_oq, pred))


def main():
    random.seed(14071)
    cleaned_df = import_data()
    text = preprocess_data(cleaned_df)
    X = vectorize_text_data(text)
    cross_validate(X, cleaned_df)


if __name__ == "__main__":
    main()
