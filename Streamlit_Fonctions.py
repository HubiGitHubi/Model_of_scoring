# import _json

import numpy as np
import streamlit as st
# from flask import jsonify
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from P7_Scoring.Features_extractions import *
from P7_Scoring.Model_extraction import *
import streamlit.components as components


def choose_id(df, my_model, cols):
    data_clients_std = StandardScaler().fit_transform(df.drop(['TARGET', 'SK_ID_CURR'], axis=1))
    data_clients_std = pd.DataFrame(data_clients_std, columns=cols)
    df['score'] = my_model.predict(data_clients_std)
    min_id, max_id = df.SK_ID_CURR.min(), df.SK_ID_CURR.max()

    id_client = st.number_input("Select the id client", min_id, max_id, value=100005)

    data_client = data_clients_std[df.SK_ID_CURR == id_client]

    if len(data_client) > 0:
        score = int(df.score[df.SK_ID_CURR == id_client])
        # score = int(my_model.predict(data_client))
    else:
        score = -1

    # st.write("0", df.SK_ID_CURR[df.score == 0][0:3].T)
    # st.write("1", df.SK_ID_CURR[df.score == 1][0:3].T)

    return score, id_client, df, data_clients_std  # jsonify(_json.load(score.to_json()))


def score_to_score_str(score: int):
    st.markdown("loan status :")
    if score == 0:
        score_str = 'accepted'
        st.success("accepted")
    elif score == 1:
        st.error("refused")
        score_str = 'refused'
    else:
        st.warning("This client is not in the database")

        score_str = 'This client is not in the database'
    return score_str


def features_importance_global(my_model, cols, df, id_client):
    try:
        feat_importance = pd.DataFrame(np.array(my_model.best_estimator_._final_estimator.feature_importances_[0]),
                                       columns=["feat_importance"])
    except:
        feat_importance = pd.DataFrame(np.array(my_model.best_estimator_._final_estimator.coef_[0]),
                                       columns=["feat_importance"])

    df_feat_importance = pd.concat([feat_importance, cols], axis=1).sort_values(by='feat_importance', ascending=False)
    df_feat_importance = df_feat_importance.set_index('Features')

    df_feat_importance['data_client'] = df[df.SK_ID_CURR == id_client][df_feat_importance.index].T
    df_feat_importance['mean_client_accepted'] = [df[col][df.score == 0].mean() for col in df_feat_importance.index]
    df_feat_importance['mean_client_refused'] = [df[col][df.score == 1].mean() for col in df_feat_importance.index]

    feat_plot = ['data_client', 'mean_client_accepted', 'mean_client_refused']

    nb_feat = 3

    df_feat_importance = df_feat_importance.reset_index()
    st.markdown('In favor of the loan :')
    for ind in df_feat_importance[0:nb_feat].index:
        st.markdown(df_feat_importance.Features[df_feat_importance.index == ind].values[0])
        st.bar_chart(df_feat_importance.loc[ind:ind, feat_plot])

    st.markdown('Against the loan :')
    for ind in df_feat_importance[-nb_feat:].index:
        st.markdown(df_feat_importance.Features[df_feat_importance.index == ind].values[0])
        st.bar_chart(df_feat_importance.loc[ind:ind, feat_plot])


def local_importance(my_model, df, data_clients_std, id_client, explainer, cols):
    with open('explainer', 'wb') as f:
        dill.dump(explainer, f)

    explanation = explainer.explain_instance(data_clients_std[df.SK_ID_CURR == id_client].values.reshape(-1),
                                             my_model.predict_proba,
                                             num_features=10)

    html_lime = explanation.as_html()
    components.v1.html(html_lime, width=900, height=350, scrolling=True)


def main():
    df, df_drop, cols = get_my_df()
    my_model = get_my_model()
    score, id_client, df, data_clients_std = choose_id(df, my_model, cols)
    score_str = score_to_score_str(score)

    if score != -1:
        explainer: object = get_my_explainer(data_clients_std, cols)
        local_importance(my_model, df, data_clients_std, id_client, explainer, cols)
        features_importance_global(my_model, cols, df, id_client)


# if__main__ == main():
main()
