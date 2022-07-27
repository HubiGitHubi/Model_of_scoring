from flask import Flask

#

# @app.route("/")
from flask import Flask, request, jsonify
import json

import _json
import dill
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from P7_Scoring.Features_extractions import *
from P7_Scoring.Model_extraction import get_my_model, get_my_explainer

app = Flask(__name__)


@app.route("/")
def loadin_is_done():
    return "Everything is loaded, let's start"


def choose_id_client(df):
    # Input for the id_client
    min_id, max_id = df.SK_ID_CURR.min(), df.SK_ID_CURR.max()
    id_client = st.number_input("Select the id client", min_id, max_id, value=100005)
    st.markdown("Not in the Database :")
    st.markdown(100005)
    st.markdown("accepted :")
    st.markdown(max_id)
    st.markdown("refused :")
    st.markdown(min_id)

    return id_client


@app.route("/")
def Calculate_all_score(df, model):
    # Calculate score for every client and store it in df
    data_clients_std = pd.DataFrame(StandardScaler().fit_transform(df.drop(['TARGET', 'SK_ID_CURR'], axis=1)),
                                    columns=df.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns)
    df['score'] = model.predict(data_clients_std)

    return data_clients_std

def
@app.route("/")
def calculate_data_client(id_client, df, data_clients_std):
    # Return the score of the chosen client. If the client is not in the dtb, return -1
    data_client = data_clients_std[df.SK_ID_CURR == id_client]

    return data_client  # jsonify(_json.load(score.to_json()))


@app.route("/")
def calculate_score_id_client(id_client, df, data_client):
    # Return the score of the chosen client. If the client is not in the dtb, return -1

    if len(data_client) > 0:
        score = int(df.score[df.SK_ID_CURR == id_client])
    else:
        score = -1

    return jsonify(_json.load(score.to_json()))


@app.route("/")
def score_to_score_str(score: int):
    # markdown the status with color : green: accepted, red: refused, yellow : not in the db
    st.markdown("loan status :")
    if score == 0:
        st.success("accepted")
    elif score == 1:
        st.error("refused")
    else:
        st.warning("This client's not in the database")


@app.route("/")
def features_importance_global(model, cols):
    try:
        feat_importance = pd.DataFrame(np.array(model.best_estimator_._final_estimator.feature_importances_[0]),
                                       columns=["feat_importance"])
    except:
        feat_importance = pd.DataFrame(np.array(model.best_estimator_._final_estimator.coef_[0]),
                                       columns=["feat_importance"])

    df_feat_importance = pd.concat([feat_importance, cols], axis=1).sort_values(by='feat_importance', ascending=False)
    df_feat_importance = df_feat_importance.set_index('Features')

    return df_feat_importance


@app.route("/")
def calcul_plot_feat_importance_glob_values(df_feat_importance, df, id_client):
    df_feat_importance['mean_clients_accepted'] = [df[col][df.score == 0].mean() for col in df_feat_importance.index]
    df_feat_importance['mean_clients_refused'] = [df[col][df.score == 1].mean() for col in df_feat_importance.index]
    df_feat_importance['data_client'] = [float(df[col][df.SK_ID_CURR == id_client].values) for col in
                                         df_feat_importance.index
                                         ]

    return df_feat_importance


@app.route("/")
def plot_feat_importance_values(df_feat_importance):
    st.markdown('Global features importance')
    st.bar_chart(df_feat_importance)


@app.route("/")
def local_importance(model, df, data_clients_std, id_client, explainer):
    id_client = st.number_input("Select the id client", 0, len(data_clients_std), value=3)

    with open('explainer', 'wb') as f:
        dill.dump(explainer, f)

    explanation = explainer.explain_instance(data_clients_std[df.SK_ID_CURR == id_client].values.reshape(-1),
                                             model.predict_proba,
                                             num_features=10)

    html_lime = explanation.as_html()
    components.html(html_lime, width=900, height=350, scrolling=True)


def main():
    df, df_drop, cols = get_my_df()
    model = get_my_model()
    id_client = choose_id_client(df)
    data_clients_std = Calculate_all_score(df, model)
    data_client = calculate_data_client(id_client, df, data_clients_std)
    score = calculate_score_id_client(id_client, df, data_client)
    score_to_score_str(score)

    if score != -1:
        df_feat_importance = features_importance_global(model, cols)
        df_feat_importance = calcul_plot_feat_importance_glob_values(df_feat_importance, df, id_client)
        plot_feat_importance_values(df_feat_importance)
        explainer = get_my_explainer(data_clients_std, cols)
        local_importance(model, df, data_clients_std, id_client, explainer)


# if__main__ == main():
main()
