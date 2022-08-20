import json
import dill
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from flask import jsonify, Flask, request


# Network URL: http: // 192.168.1.27:8501

def get_my_model() -> object:
    # Charge the best model

    with open('Datas/best_model', 'rb') as f1:
        my_model = pickle.load(f1)

    return my_model


def get_my_explainer():
    # Charge the explainer'

    with open('Datas/explainer', 'rb') as f:
        explainer = pickle.load(f, errors="ignore")

    return explainer


def get_train_test() -> object:
    path = 'Datas/data_clients_sampled.csv'
    df = pd.read_csv(path)
    path = 'Datas/data_clients_to_predict.csv'
    df_to_predict = pd.read_csv(path)

    return df, df_to_predict


def Calculate_all_scores(df_to_predict, model):
    # Calculate score for every client and store it in df
    # The client is considerate negative if the model predicts 60% negative or more
    data_clients_std = pd.DataFrame(StandardScaler().fit(df.drop(['SK_ID_CURR', 'TARGET'], axis=1)).transform(
        df_to_predict.drop(['SK_ID_CURR'], axis=1)),
        columns=df_to_predict.drop(['SK_ID_CURR'], axis=1).columns)

    x_test_predict = pd.DataFrame(model.predict_proba(data_clients_std.values), columns=['positive', 'negative'])
    x_test_predict['pred'] = -1
    x_test_predict['pred'][x_test_predict['positive'] >= .6] = 1
    x_test_predict['pred'][x_test_predict['positive'] < .6] = 0
    df_to_predict['score'] = x_test_predict['pred'].values

    return data_clients_std


def calculate_score_id_client(id_client, df_to_predict, data_client):
    # Return the score of the chosen client. If the client is not in the dtb, return -1

    if len(data_client) > 0:
        score = int(df_to_predict.score[df_to_predict.SK_ID_CURR == id_client])
    else:
        score = -1

    return score


def predict_proba_client(data_client, model):
    # Return proba of success/failure of a client
    proba_client = model.predict_proba(data_client)
    return proba_client


# noinspection PyProtectedMember
def features_importance_global(model, df):
    # Calculate the global features importance
    try:
        feat_importance = pd.DataFrame(np.array(model.best_estimator_._final_estimator.feature_importances_[0]),
                                       columns=["feat_importance"])
    except:
        feat_importance = pd.DataFrame(np.array(model.best_estimator_._final_estimator.coef_[0]),
                                       columns=["feat_importance"])
    cols = pd.DataFrame(df.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns, columns=['Features'])
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
df, df_to_predict = get_train_test()
model = get_my_model()
data_clients_std = Calculate_all_scores(df_to_predict, model)
df_feat_importance = features_importance_global(model, df)
explainer = get_my_explainer()

app = Flask(__name__)


@app.route("/")
def beginning():
    return "Model and data are now loaded"


@app.route('/get_df_values/')
def get_df() -> object:
    df_json = json.loads(df.to_json())

    return jsonify(df_json)


@app.route('/get_df_predict_values/')
def get_df_to_predict() -> object:
    df_to_predict_json = json.loads(df_to_predict.to_json())

    return jsonify(df_to_predict_json)


@app.route('/Calculate_all_datas_values/')
def Calculate_all_datas():
    data_clients_std_json = json.loads(data_clients_std.to_json())
    return jsonify(data_clients_std_json)


@app.route('/calculate_data_client_values/')
def calculate_data_client_std():
    # Return the data of the chosen client
    id_client = int(request.args.get('id_client'))
    data_client_json = json.loads(df_to_predict[df_to_predict.SK_ID_CURR == id_client].to_json())

    return jsonify(data_client_json)


@app.route('/calculate_score_id_client_values/')
def calculate_score_id_client():
    # Return the score of the chosen client. If the client is not in the dtb, return -1
    id_client = int(request.args.get('id_client'))
    st.write(df_to_predict.score[df_to_predict.SK_ID_CURR == id_client])

    if len(df_to_predict[df_to_predict.SK_ID_CURR == id_client]) > 0:
        score = int(df_to_predict.score[df_to_predict.SK_ID_CURR == id_client])
        st.write('score with len min1', score)
    else:
        score = -1
    score_json = score

    return jsonify(score_json)


@app.route('/predict_proba_client_values/')
def predict_proba_client():
    # Return proba of success/failure of a client
    id_client = int(request.args.get('id_client'))
    data_client = data_clients_std[df_to_predict.SK_ID_CURR == id_client].values
    proba_client_json = model.predict_proba(data_client).tolist()[0]

    return jsonify(proba_client_json)


@app.route('/features_importance_global_values/')
def features_importance_global():
    # Calculate the global features importance
    df_feat_importance_json = json.loads(df_feat_importance.to_json())

    return jsonify(df_feat_importance_json)


@app.route('/local_importance_values/')
def local_importance_explanation_list():
    id_client = int(request.args.get('id_client'))
    nb_feats = int(request.args.get('nb_feats'))
    data_client = data_clients_std[df_to_predict.SK_ID_CURR == id_client]
    explanation = explainer.explain_instance(data_client.values.reshape(-1),
                                             model.predict_proba,
                                             num_features=nb_feats)
    explanation_json = explanation.as_list()
    return jsonify(explanation_json)


@app.route('/find_loc_feat_importance_values/')
def find_loc_feat_importance():
    # Return the name of most important locale features
    id_client = int(request.args.get('id_client'))
    nb_feats = int(request.args.get('nb_feats'))
    data_client = data_clients_std[df_to_predict.SK_ID_CURR == id_client]
    explanation = explainer.explain_instance(data_client.values.reshape(-1),
                                             model.predict_proba,
                                             num_features=nb_feats).as_list()
    liste = []
    final_list = []

    for i in explanation:
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

    return jsonify(final_list)


@app.route('/find_loc_feat_importance_values_as_list/')
def find_loc_feat_importance_as_list():
    # Return the name of most important locale features
    id_client = int(request.args.get('id_client'))
    nb_feats = int(request.args.get('nb_feats'))
    data_client = data_clients_std[df_to_predict.SK_ID_CURR == id_client]
    explanation = explainer.explain_instance(data_client.values.reshape(-1),
                                             model.predict_proba,
                                             num_features=nb_feats).as_list()
    return jsonify(explanation)


if __name__ == "__main__":
    app.run()
