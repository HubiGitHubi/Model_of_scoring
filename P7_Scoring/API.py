import json
import math
import dill
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib import pyplot as plt

# Local URL: http: // localhost: 8501
# Network URL: http: // 192.168.1.27:8501

# __________________________________________________________________________________________________
# Importation functions
API_URL = "http://192.168.1.27:8501/"


#@st.cache
#def import_side_bar():
    # return the setting of the settings page
    # URL of the API + function
#   URL = API_URL + "side_bar_values/"
    # Asks the variable to the API and store it
#    response = requests.get(URL)
    # JSON to Python
#   content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
#   id_client = pd.Series(content['SK_ID_CURR'])
#   yes_no_feat_glob = pd.Series(content['yes_no_feat_glob'])
#   nb_feats = pd.Series(content['nb_feats'])
#  return id_client, yes_no_feat_glob, nb_feats

@st.cache
def add_side_bar(df_to_predict):
    # Add the page with settings and store the settings

    min_id, max_id = df_to_predict.SK_ID_CURR.min(), df_to_predict.SK_ID_CURR.max()
    id_client = st.sidebar.number_input("Select the id client", min_id, max_id)

    yes_no_feat_glob = st.sidebar.selectbox(
        "Do you want the global features importance ? : ",
        ("Yes", "No"))
    nb_feats = st.sidebar.slider(
        "How many local features do you want ?", 2, 15, step=1)

    return id_client,yes_no_feat_glob, nb_feats

@st.cache
def get_my_model() -> object:
    # return my_model

    # URL of the API + function
    URL = API_URL + "get_my_model_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    my_model = content['my_model']

    return my_model


@st.cache
def get_my_explainer():
    # Charge the explainer'

    # URL of the API + function
    URL = API_URL + "get_my_explainer_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    explainer = content['explainer']

    return explainer


@st.cache
def Import_train_test() -> object:
    # URL of the API + function
    URL = API_URL + "get_train_test_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content)
    # Transforms in Pandas
    df = pd.DataFrame(content['df'])
    df_drop = pd.DataFrame(content['df_drop'])
    cols = pd.DataFrame(['cols'])
    df_to_predict = pd.DataFrame(content['df_to_predict'])

    return df, df_drop, cols, df_to_predict


@st.cache
def Import_all_scores():
    # URL of the API + function
    URL = API_URL + "Calculate_all_scores_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    data_clients_std = pd.DataFrame(content["data_clients_std"])
    data_clients_std_train = pd.DataFrame(content["data_clients_std_train"])
    return data_clients_std, data_clients_std_train


@st.cache
def Import_data_client():
    # Return the data of the chosen client

    # URL of the API + function
    URL = API_URL + "calculate_data_client_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    data_client = pd.DataFrame(content["data_client"])

    return data_client


@st.cache
def Import_score_id_client():
    # Return the score of the chosen client. If the client is not in the dtb, return -1

    # URL of the API + function
    URL = API_URL + "calculate_data_client_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    score = content["score"].values
    return score


@st.cache
def Import_proba_client():
    # Return proba of success/failure of a client

    # URL of the API + function
    URL = API_URL + "predict_proba_client_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    proba_client = content["proba_client"]
    return proba_client


@st.cache
def Import_features_importance_global():
    # Calculate the global features importance

    # URL of the API + function
    URL = API_URL + "features_importance_global_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    df_feat_importance = pd.DataFrame(content["df_feat_importance"])

    return df_feat_importance


@st.cache
def Import_local_importance():
    # URL of the API + function
    URL = API_URL + "local_importance_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    explanation_list = pd.Series(content["df_feat_importance"])

    return explanation_list


@st.cache
def Import_find_loc_feat_importance():
    # Return the name of most important locale features

    # URL of the API + function
    URL = API_URL + "find_loc_feat_importance_values/"
    # Asks the variable to the API and store it
    response = requests.get(URL)
    # JSON to Python
    content = json.loads(response.content.decode('utf-8'))
    # Transforms in Pandas
    final_list = pd.Series(content["final_list"])
    return final_list


# __________________________________________________________________________________________________
# Plot Functions
@st.cache
def plot_proba_client(proba_client):
    # Plot the proba client
    st.write("Repayment rate")
    st.success(round(proba_client[0][0], 2))
    st.write("Default rate")
    st.error(round(proba_client[0][1], 2))


@st.cache
def score_to_score_str(score: int):
    # markdown the status with color : green: accepted, red: refused, yellow : not in the db
    st.markdown("loan status :")
    if score == 0:
        st.success("accepted")
    elif score == 1:
        st.error("refused")
    else:
        st.warning("This client's not in the database")


@st.cache
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


@st.cache
def plot_feat_importance_values(df_feat_importance):
    # Plot the global features importance
    df_feat_importance = df_feat_importance.sort_values(by='feat_importance', ascending=False)
    st.write("Global feature importance")
    st.bar_chart(df_feat_importance, height=500)


@st.cache
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


@st.cache
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


@st.cache
def hist_feats_loc(final_list, nb_feats, df_to_predict):
    # Plot the number of chosen local most important feats

    _ = math.ceil(math.sqrt(len(final_list)))
    if nb_feats // _ == nb_feats / _:
        nb_cols = nb_feats // _
    else:
        nb_cols = nb_feats // _ + 1

    fig, axs = plt.subplots(_, nb_cols, sharey=True)

    for i, _c in enumerate(final_list):
        ax = axs.flat[i]
        ax.hist(df_to_predict[[_c]], bins=20)
        ax.set_title(_c)
        fig.set_tight_layout(h_pad=len(_c))
    st.pyplot(fig)


def main():
    df, df_drop, cols, df_to_predict = Import_train_test()
    id_client, yes_no_feat_glob, nb_feats = add_side_bar(df_to_predict)
    model = get_my_model()
    data_clients_std, data_clients_std_train = Import_all_scores(df_to_predict, df_drop, model)
    data_client = Import_data_client(id_client, df_to_predict, data_clients_std)
    score = Import_score_id_client(id_client, df_to_predict, data_client)
    score_to_score_str(score)
    df_feat_importance = features_importance_global(model, cols)

    if score != -1:

        if yes_no_feat_glob == 'Yes':
            plot_feat_importance_values(df_feat_importance)
        proba_client = Import_proba_client(data_client, model)
        plot_proba_client(proba_client)
        explainer = get_my_explainer(data_clients_std, cols)
        explanation_list = local_importance(model, data_client, explainer, nb_feats)
        if nb_feats > 0:
            final_list = find_loc_feat_importance(explanation_list, df_to_predict)
            hist_feats_loc(final_list, nb_feats, df_to_predict)


# if__main__ == main():
main()
