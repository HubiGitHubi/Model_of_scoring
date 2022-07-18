# import _json

import numpy as np
import streamlit as st
# from flask import jsonify
#import sklearn
from sklearn.preprocessing import StandardScaler
from P7_Scoring.Features_extractions import *
from P7_Scoring.Model_extraction import *
import streamlit.components.v1 as components


def choose_id_client(df):
    # Input for the id_client
    min_id, max_id = df.SK_ID_CURR.min(), df.SK_ID_CURR.max()
    id_client = st.number_input("Select the id client", min_id, max_id, value=100005)

    return id_client


def Calculate_all_score(df, model):
    # Calculate score for every client and store it in df
    data_clients_std = pd.DataFrame(StandardScaler().fit_transform(df.drop(['TARGET', 'SK_ID_CURR'], axis=1)),
                                    columns=df.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns)
    df['score'] = model.predict(data_clients_std)

    return data_clients_std


def calculate_score_id_client(id_client, df, data_clients_std):
    # Return the score of the chosen client. If the client is not in the dtb, return -1
    data_client = data_clients_std[df.SK_ID_CURR == id_client]

    if len(data_client) > 0:
        score = int(df.score[df.SK_ID_CURR == id_client])
    else:
        score = -1

    return score, data_client  # jsonify(_json.load(score.to_json()))


def score_to_score_str(score: int):
    # markdown the status with color : green: accepted, red: refused, yellow : not in the db
    st.markdown("loan status :")
    if score == 0:
        st.success("accepted")
    elif score == 1:
        st.error("refused")
    else:
        st.warning("This client's not in the database")


# noinspection PyProtectedMember
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


def calcul_plot_feat_importance_glob_values(df_feat_importance, df, id_client):
    df_feat_importance['mean_clients_accepted'] = [df[col][df.score == 0].mean() for col in df_feat_importance.index]
    df_feat_importance['mean_clients_refused'] = [df[col][df.score == 1].mean() for col in df_feat_importance.index]
    df_feat_importance['data_client'] = [float(df[col][df.SK_ID_CURR == id_client].values) for col in
                                         df_feat_importance.index
                                         ]

    return df_feat_importance


def plot_feat_importance_values(df_feat_importance):
    feat_plot = ['data_client', 'mean_clients_accepted', 'mean_clients_refused']
    nb_feat = 3
    width = 2
    height = 200
    df_feat_importance = df_feat_importance.reset_index()
    st.markdown('In favor of the loan :')
    for ind in df_feat_importance[0:nb_feat].Features:
        st.markdown(ind)
        st.bar_chart(df_feat_importance[df_feat_importance.Features == str(ind)][feat_plot].T,
                     width=width,
                     height=height)

    st.markdown('Against the loan :')
    for ind in df_feat_importance[-nb_feat:].Features:
        st.markdown(ind)
        st.bar_chart(df_feat_importance[df_feat_importance.Features == ind][feat_plot].T,
                     width=width,
                     height=height)


def local_importance(model, df, data_clients_std, id_client, explainer):
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
    score, data_client = calculate_score_id_client(id_client, df, data_clients_std)
    score_to_score_str(score)

    if score != -1:
        df_feat_importance = features_importance_global(model, cols)
        df_feat_importance = calcul_plot_feat_importance_glob_values(df_feat_importance, df, id_client)
        plot_feat_importance_values(df_feat_importance)
        explainer = get_my_explainer(data_clients_std, cols)
        local_importance(model, df, data_clients_std, id_client, explainer)


# if__main__ == main():
main()