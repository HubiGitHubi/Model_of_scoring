import pickle
import pandas as pd


# import imblearn *

def get_my_model() -> object:
    """

    :rtype: object
    """
    # Charger le best model
    # with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/best_model', 'rb') as f1:
    # my_model = pickle.load(f1)
    # return my_model

    try:
        with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/best_model', 'rb') as f1:
            my_model = pickle.load(f1)
    except:
        with open('Datas/best_model', 'rb') as f1:
            my_model = pickle.load(f1)
    return my_model


def get_my_explainer():
    # Charge the explainer'
    try:
        with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    except:
        with open('Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    return explainer


def get_train_test() -> object:
    try:
        path = 'C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/data_clients.csv'
        df = pd.read_csv(path)
    except:
        path = 'Datas/data_clients.csv'
        df = pd.read_csv(path)
    try:
        path = 'C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/data_clients_to_predict.csv'
        df_to_predict = pd.read_csv(path)
    except:
        path = 'Datas/data_clients_to_predict.csv'
        df_to_predict = pd.read_csv(path)

    df_drop = df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    cols = pd.DataFrame(df_drop.columns, columns=['Features'])

    return df, df_drop, cols, df_to_predict
