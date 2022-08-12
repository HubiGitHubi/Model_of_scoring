import json

import dill
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

from flask import jsonify, Flask, request

'''
cd  C:/Users/33646/Documents/OpenClassroom/'Projet 7'/Model_of_scoring
$env:FLASK_APP = "app"
flask run'''


# Network URL: http: // 192.168.1.27:8501

def get_my_model() -> object:
    """

    :rtype: object
    """
    # Charger le best model
    # with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/best_model', 'rb') as f1:
    # my_model = pickle.load(f1)
    # return my_model

    try:
        with open('Datas/best_model', 'rb') as f1:
            my_model = pickle.load(f1)
    except:
        with open('Datas/best_model', 'rb') as f1:
            my_model = pickle.load(f1)
    return my_model


def get_my_explainer():
    # Charge the explainer'
    try:
        with open('Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    except:
        with open('Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    return explainer


def get_train_test() -> object:
    # try:
    path = 'Datas/data_clients.csv'
    df = pd.read_csv(path).sample(frac=.5)
    # except:
    # path = 'C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/data_clients.csv'
    # df = pd.read_csv(path)
    # try:
    path = 'Datas/data_clients_to_predict.csv'
    df_to_predict = pd.read_csv(path)
    # except:
    #   path = 'C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/data_clients_to_predict.csv'
    #   df_to_predict = pd.read_csv(path)

    df_drop = df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    cols = pd.DataFrame(df_drop.columns, columns=['Features'])
    df_full = pd.concat([df, df_to_predict], axis=1)
    return df, df_drop, cols, df_to_predict, df_full


def Calculate_all_scores(df_to_predict, df_drop, model):
    # Calculate score for every client and store it in df
    data_clients_std_train = pd.DataFrame(StandardScaler().fit(df_drop).transform(df_drop), columns=df_drop.columns)
    data_clients_std = pd.DataFrame(StandardScaler().fit(df_drop).transform(df_to_predict.drop(['SK_ID_CURR'], axis=1)),
                                    columns=df_drop.columns)
    df_to_predict['score'] = model.predict(data_clients_std.values)
    return data_clients_std, data_clients_std_train


def calculate_data_client(id_client, df_to_predict, data_clients_std):
    # Return the data of the chosen client

    data_client = data_clients_std[df_to_predict.SK_ID_CURR == id_client]
    return data_client  # jsonify(_json.load(score.to_json()))


def calculate_score_id_client(id_client, df_to_predict, data_client):
    # Return the score of the chosen client. If the client is not in the dtb, return -1

    if len(data_client) > 0:
        score = int(df_to_predict.score[df_to_predict.SK_ID_CURR == id_client])
    else:
        score = -1

    return score  # jsonify(_json.load(score.to_json()))


def predict_proba_client(data_client, model):
    # Return proba of success/failure of a client
    proba_client = model.predict_proba(data_client)
    return proba_client


# noinspection PyProtectedMember
def features_importance_global(model, cols):
    # Calculate the global features importance
    try:
        feat_importance = pd.DataFrame(np.array(model.best_estimator_._final_estimator.feature_importances_[0]),
                                       columns=["feat_importance"])
    except:
        feat_importance = pd.DataFrame(np.array(model.best_estimator_._final_estimator.coef_[0]),
                                       columns=["feat_importance"])

    df_feat_importance = pd.concat([feat_importance, cols], axis=1).sort_values(by='feat_importance', ascending=False)
    df_feat_importance = df_feat_importance.set_index('Features')

    return df_feat_importance


def local_importance(model, data_client, explainer, nb_feats):
    with open('explainer', 'wb') as f:
        dill.dump(explainer, f)
    explanation = explainer.explain_instance(data_client.values.reshape(-1),
                                             model.predict_proba,
                                             num_features=nb_feats)

    explanation_list = explanation.as_list()
    with plt.style.context("ggplot"):
        st.pyplot(explanation.as_pyplot_figure())
    return explanation_list


def find_loc_feat_importance(explanation_list, df_to_predict):
    # Return the name of most important locale features
    liste = []
    final_list = []

    for i in explanation_list:
        if ">" in i[0]:
            symbol = '>'
        else:
            symbol = '<'
        liste.append(i[0].split(symbol)[0][0:-1])

    for i in liste:
        try:
            aa = df_to_predict[i]
            final_list.append(i)
        except:
            a = 1
    return final_list


# Import data, model, explainer
df, df_drop, cols, df_to_predict, df_full = get_train_test()
model = get_my_model()
data_clients_std, data_clients_std_train = Calculate_all_scores(df_to_predict, df_drop, model)
df_feat_importance = features_importance_global(model, cols)
explainer = get_my_explainer()

app = Flask(__name__)


@app.route("/")
def beginning():
    return "Model and data are now loaded"


# Putting ids in dictionary (json file)
@app.route('/ids/')
def ids_list():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    customers_id_list = df["SK_ID_CURR"].sort_values()
    # Convert Series to JSON
    customers_id_list_json = json.loads(customers_id_list.to_json())
    # Returning the processed data
    # jsonify is a helper method provided by Flask to properly return JSON data.
    return jsonify(customers_id_list_json)


@app.route('/get_df_values/')
def get_df() -> object:
    df_json = json.loads(df.to_json())

    return jsonify(df_json)


@app.route('/get_cols_values/')
def get_cols() -> object:
    cols_json = json.loads(cols.to_json())

    return jsonify(cols_json)


@app.route('/get_df_drop_values/')
def get_df_drop() -> object:
    df_drop_json = json.loads(df_drop.to_json())

    return jsonify(df_drop_json)


@app.route('/get_df_predict_values/')
def get_df_to_predict() -> object:
    df_to_predict_json = json.loads(df_to_predict.to_json())

    return jsonify(df_to_predict_json)


@app.route('/Calculate_all_scores_values')
def Calculate_all_scores(data_clients_std, data_clients_std_train):
    data_clients_std_json = json.loads(data_clients_std.to_json())
    data_clients_std_train_json = json.loads(data_clients_std_train.to_json())
    return jsonify(data_clients_std_json, data_clients_std_train_json)


@app.route('/calculate_data_client_values')
def calculate_data_client():
    # Return the data of the chosen client
    id_client = int(request.args.get('SK_ID_CURR'))
    data_client = data_clients_std[df_to_predict.SK_ID_CURR == id_client]
    data_client_json = json.loads(data_client.to_json())

    return jsonify(data_client_json)


@app.route('/calculate_score_id_client_values')
def calculate_score_id_client(df_to_predict, data_client):
    # Return the score of the chosen client. If the client is not in the dtb, return -1
    id_client = int(request.args.get('SK_ID_CURR'))

    if len(data_client) > 0:
        score = int(df_to_predict.score[df_to_predict.SK_ID_CURR == id_client])
    else:
        score = -1
    score_json = json.loads(score.to_json())

    return jsonify(score_json)


@app.route('/predict_proba_client_values')
def predict_proba_client(data_client, model):
    # Return proba of success/failure of a client
    proba_client = model.predict_proba(data_client)
    proba_client_json = json.loads(proba_client.to_json())

    return jsonify(proba_client_json)


@app.route('/features_importance_global_values')
def features_importance_global(df_feat_importance):
    # Calculate the global features importance
    df_feat_importance_json = json.loads(df_feat_importance.to_json())

    return jsonify(df_feat_importance_json)


@app.route('/local_importance_values')
def local_importance(model, data_client, explainer, nb_feats):
    explanation = explainer.explain_instance(data_client.values.reshape(-1),
                                             model.predict_proba,
                                             num_features=nb_feats)
    explanation_list = explanation.as_list()
    explanation_list_json = json.loads(explanation_list.to_json())
    explanation_json = json.loads(explanation_list.to_json())
    return jsonify(explanation_list_json, explanation_json)


@app.route('/find_loc_feat_importance_values')
def find_loc_feat_importance(explanation_list, df_to_predict):
    # Return the name of most important locale features
    liste = []
    final_list = []

    for i in explanation_list:
        if ">" in i[0]:
            symbol = '>'
        else:
            symbol = '<'
        liste.append(i[0].split(symbol)[0][0:-1])

    for i in liste:
        try:
            a = df_to_predict[i]
            final_list.append(i)
        except:
            a = 1
    final_list_json = json.loads(final_list.to_json())

    return jsonify(final_list_json)


if __name__ == "__main__" :
    app.run()
