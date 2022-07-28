# import _json

import dill
import numpy as np
import streamlit as st
from flask import jsonify, Flask
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from P7_Scoring.Extraction import *

# Local URL: http: // localhost: 8501
# Network URL: http: // 192.168.1.27:8501
Flask_values = Flask(__name__)


@Flask_values.route("/Import_and_calcul_functions/side_bar_values")
def add_side_bar(df_to_predict):
    # Add the page with settings and store the settings

    min_id, max_id = df_to_predict.SK_ID_CURR.min(), df_to_predict.SK_ID_CURR.max()
    id_client = st.sidebar.number_input("Select the id client", min_id, max_id)

    yes_no_feat_glob = st.sidebar.selectbox(
        "Do you want the global features importance ? : ",
        ("Yes", "No"))
    nb_feats = st.sidebar.slider(
        "How many local features do you want ?", 2, 15, step=1)

    return jsonify({'SK_ID_CURR': id_client,
                    'yes_no_feat_glob': yes_no_feat_glob,
                    'nb_feats': nb_feats})

@Flask_values.route("/Import_and_calcul_functions/get_my_model_values")
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
    return jsonify({'my_model': my_model})


@Flask_values.route("/Import_and_calcul_functions/get_my_explainer_values")
def get_my_explainer():
    # Charge the explainer'
    try:
        with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    except:
        with open('Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    return jsonify({'explainer': explainer})


@Flask_values.route("/Import_and_calcul_functions/get_train_test_values")
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

    return jsonify({'df': df,
                    'df_drop': df_drop,
                    'cols': cols,
                    'df_to_predict': df_to_predict})


@Flask_values.route("/Import_and_calcul_functions/Calculate_all_scores_values")
def Calculate_all_scores(df_to_predict, df_drop, model):
    # Calculate score for every client and store it in df
    data_clients_std_train = pd.DataFrame(StandardScaler().fit(df_drop).transform(df_drop), columns=df_drop.columns)
    data_clients_std = pd.DataFrame(StandardScaler().fit(df_drop).transform(df_to_predict.drop(['SK_ID_CURR'], axis=1)),
                                    columns=df_drop.columns)
    df_to_predict['score'] = model.predict(data_clients_std.values)
    return jsonify({'data_clients_std': data_clients_std,
                    'data_clients_std_train': data_clients_std_train})


@Flask_values.route("/Import_and_calcul_functions/calculate_data_client_values")
def calculate_data_client(id_client, df_to_predict, data_clients_std):
    # Return the data of the chosen client
    data_client = data_clients_std[df_to_predict.SK_ID_CURR == id_client]
    return jsonify({'data_client': data_client})


@Flask_values.route("/Import_and_calcul_functions/calculate_score_id_client_values")
def calculate_score_id_client(id_client, df_to_predict, data_client):
    # Return the score of the chosen client. If the client is not in the dtb, return -1
    if len(data_client) > 0:
        score = int(df_to_predict.score[df_to_predict.SK_ID_CURR == id_client])
    else:
        score = -1

    return jsonify({'score': score})  # jsonify(_json.load(score.to_json()))


@Flask_values.route("/Import_and_calcul_functions/predict_proba_client_values")
def predict_proba_client(data_client, model):
    # Return proba of success/failure of a client
    proba_client = model.predict_proba(data_client)
    return jsonify({'proba_client': proba_client})


@Flask_values.route("/Import_and_calcul_functions/features_importance_global_values")
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

    return jsonify({'df_feat_importance': df_feat_importance})


@Flask_values.route("/Import_and_calcul_functions/local_importance_values")
def local_importance(model, data_client, explainer, nb_feats):
    with open('explainer', 'wb') as f:
        dill.dump(explainer, f)
    explanation = explainer.explain_instance(data_client.values.reshape(-1),
                                             model.predict_proba,
                                             num_features=nb_feats)

    explanation_list = explanation.as_list()
    with plt.style.context("ggplot"):
        st.pyplot(explanation.as_pyplot_figure())
    return jsonify({'explanation_list': explanation_list})


@Flask_values.route("/Import_and_calcul_functions/find_loc_feat_importance_values")
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
    return jsonify({'final_list': final_list})


def main():
    df, df_drop, cols, df_to_predict = get_train_test()
    id_client, yes_no_feat_glob, nb_feats = add_side_bar(df_to_predict)
    model = get_my_model()
    data_clients_std, data_clients_std_train = Calculate_all_scores(df_to_predict, df_drop, model)
    data_client = calculate_data_client(id_client, df_to_predict, data_clients_std)
    score = calculate_score_id_client(id_client, df_to_predict, data_client)
    df_feat_importance = features_importance_global(model, cols)

    if score != -1:

        if yes_no_feat_glob == 'Yes':
            proba_client = predict_proba_client(data_client, model)
        explainer = get_my_explainer()
        explanation_list = local_importance(model, data_client, explainer, nb_feats)
        final_list = find_loc_feat_importance(explanation_list, df_to_predict)


if __name__ == main():
    Flask_values.run()
