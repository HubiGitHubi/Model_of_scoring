import math
import dill
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import seaborn as sns


# Local URL: http: // localhost: 8501
# Network URL: http: // 192.168.1.27:8501
def id_client_side_bar():
    # Add the page with settings and store the settings

    id_client = st.sidebar.number_input("Select the id client", 100001, 456250)
    return id_client


def yes_no_feat_glob_side_bar():
    yes_no_feat_glob = st.sidebar.selectbox(
        "Do you want the global features importance ? : ",
        ("Yes", "No"))
    return yes_no_feat_glob


def nb_feats_side_bar():
    nb_feats = st.sidebar.slider(
        "How many local features do you want ?", 2, 15, step=1)
    return nb_feats


def nb_neighbours():
    nb_neighbours = st.sidebar.slider(
        "How many local neighbours do you want ?", 2, 15, step=1)
    return nb_neighbours


def multi_choice_neighbours():
    options = st.sidebar.multiselect(
        'What kind of K neighbours graph do you want ?',
        ['Positive clients (1)', 'Negatives clients (0)', 'all clients mixed (1&0)'], ['all clients mixed (1&0)'])

    return options


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
        with open('../Datas/best_model', 'rb') as f1:
            my_model = pickle.load(f1)
    return my_model


def number_of_neighbours():
    nb_neighbours = st.sidebar.slider(
        "How many local neighbours do you want ?", 10, 456250, step=1)
    return nb_neighbours


def get_my_explainer():
    # Charge the explainer'
    try:
        with open('Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    except:
        with open('../Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    return explainer


def get_train_test() -> object:
    # try:
    path = 'Datas/data_clients.csv'
    df = pd.read_csv(path)
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

    return df, df_drop, cols, df_to_predict


def Calculate_all_scores(df_to_predict, df_drop, model):
    # Calculate score for every client and store it in df
    data_clients_std_train = pd.DataFrame(StandardScaler().fit(df_drop).transform(df_drop), columns=df_drop.columns)
    data_clients_std = pd.DataFrame(StandardScaler().fit(df_drop).transform(df_to_predict.drop(['SK_ID_CURR'], axis=1)),
                                    columns=df_drop.columns)
    df_to_predict['score'] = model.predict(data_clients_std.values)
    return data_clients_std, data_clients_std_train


def calculate_data_client( id_client, df_to_predict, data_clients_std ):
    # Return the data of the chosen client
    data_client_std = data_clients_std[df_to_predict.SK_ID_CURR == id_client]
    data_client = df_to_predict[df_to_predict.SK_ID_CURR == id_client]

    return data_client_std,data_client  # jsonify(_json.load(score.to_json()))


def calculate_score_id_client( id_client, df_to_predict, data_client_std ):
    # Return the score of the chosen client. If the client is not in the dtb, return -1

    if len(data_client_std) > 0:
        score = int(df_to_predict.score[df_to_predict.SK_ID_CURR == id_client])
    else:
        score = -1

    return score  # jsonify(_json.load(score.to_json()))


def predict_proba_client( data_client_std, model ):
    # Return proba of success/failure of a client
    proba_client = model.predict_proba(data_client_std)
    return proba_client


def plot_proba_client( proba_client ):
    # Plot the proba client
    st.write("Repayment rate")
    st.success(round(proba_client[0][0], 2))
    st.write("Default rate")
    st.error(round(proba_client[0][1], 2))


def score_to_score_str( score: int ):
    # markdown the status with color : green: accepted, red: refused, yellow : not in the db
    st.markdown("loan status :")
    if score == 0:
        st.success("accepted")
    elif score == 1:
        st.error("refused")
    else:
        st.warning("This client's not in the database")


# noinspection PyProtectedMember
def features_importance_global( model, cols ):
    # Calculate the global features importance and filter only the 15 most important (positive or negative)
    try:
        feat_importance = pd.DataFrame(np.array(model.best_estimator_._final_estimator.feature_importances_[0]),
                                       columns=["feat_importance"])
    except:
        feat_importance = pd.DataFrame(np.array(model.best_estimator_._final_estimator.coef_[0]),
                                       columns=["feat_importance"])

    df_feat_importance = pd.concat([feat_importance, cols], axis=1).sort_values(by='feat_importance', ascending=False)
    df_feat_importance = df_feat_importance.set_index('Features')
    df_feat_importance['abs'] = abs(df_feat_importance['feat_importance'])
    df_feat_importance = df_feat_importance.sort_values(by='abs', ascending=False).reset_index()[0:15]
    df_plot = df_feat_importance.loc[0:15, ['Features', 'feat_importance']]

    return df_plot


def plot_feat_importance_values( df_plot ):
    # Plot the global features importance
    st.write("Global feature importance")
    fig = plt.figure(figsize=(15, 25))
    sns.barplot(data=df_plot.reset_index().sort_values(by='feat_importance', ascending=False),
                x="feat_importance", y='Features')
    st.write(fig)


def local_importance( model, data_client_std, explainer, nb_feats ):
    with open('../explainer', 'wb') as f:
        dill.dump(explainer, f)
    explanation = explainer.explain_instance(data_client_std.values.reshape(-1),
                                             model.predict_proba,
                                             num_features=nb_feats)

    explanation_list = explanation.as_list()
    with plt.style.context("ggplot"):
        st.pyplot(explanation.as_pyplot_figure())
    return explanation_list


def find_loc_feat_importance( explanation_list, df_to_predict ):
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


def hist_feats_loc( final_list, nb_feats, df_to_predict, data_client ):
    # Plot the number of chosen local most important feats

    _ = math.ceil(math.sqrt(len(final_list)))
    if nb_feats//_ == nb_feats/_:
        nb_cols = nb_feats//_
    else:
        nb_cols = nb_feats//_+1

    fig, axs = plt.subplots(_, nb_cols, sharey=True)

    for i, _c in enumerate(final_list):
        ax = axs.flat[i]
        ax.hist(df_to_predict[[_c]], bins=20)
        ax.axvline(data_client[_c][0], color='red')
        ax.legend({'The client', 'Other clients'})
        ax.set_title(_c)
        fig.set_tight_layout(True)
    st.pyplot(fig)

    # ______________________________________________________________________________________________________________________
    # find 20 nearest neighbors among the training set


def Calculate_neighbourhood(df, df_to_predict, nb_neighbours, final_list,data_client):

    # return the closest neighbors final feats list (nb_neighbours chosen by the user)
    neighbors = NearestNeighbors(n_neighbors=nb_neighbours).fit(df.drop(['SK_ID_CURR', 'TARGET'], axis=1))

    index_neighbors = neighbors.kneighbors(X=data_client.drop(['SK_ID_CURR', 'score'], axis=1).values,
                                           n_neighbors=nb_neighbours, return_distance=False).ravel()

    #index_neighbors = neighbors.kneighbors_graph([df_to_predict['SK_ID_CURR']]).indices

    st.write(index_neighbors)

    neighbors = df.loc[index_neighbors.index, final_list]
    st.write(neighbors)
    return neighbors

"""
def Calculate_neighbourhood_positive( df, df_to_predict, nb_neighbours, final_list ):
    df_pos = df[df["TARGET"] == 1]
    df_to_predict_pos = df_to_predict[df_to_predict["score"] == 1]

    # return the closest neighbors final feats list (nb_neighbours chosen by the user)
    neighbors_pos = NearestNeighbors(n_neighbors=nb_neighbours).fit(df_pos.drop(['TARGET'], axis=1))
    index_neighbors = neighbors_pos.kneighbors(X=df_to_predict_pos.drop(['score'], axis=1),
                                               n_neighbors=nb_neighbours).ravel()
    neighbors_pos = df_pos.loc[index_neighbors, final_list]
    return neighbors_pos


def Calculate_neighbourhood_negative( df, df_to_predict, nb_neighbours, final_list ):
    df_neg = df[df["TARGET"] == 0]
    df_to_predict_neg = df_to_predict[df_to_predict["score"] == 0]

    # return the closest neighbors final feats list (nb_neighbours chosen by the user)
    neighbors_neg = NearestNeighbors(n_neighbors=nb_neighbours).fit(df_neg.drop(['TARGET'], axis=1))
    index_neighbors = neighbors_neg.kneighbors(X=df_to_predict_neg.drop(['score'], axis=1),
                                               n_neighbors=nb_neighbours).ravel()
    neighbors_neg = df_neg.loc[index_neighbors, final_list]
    return neighbors_neg
"""

def plot_neigh( neighbors, final_list, nb_feats ):
    # Plot local most important feats for the number of chosen neighbours

    _ = math.ceil(math.sqrt(len(final_list)))
    if nb_feats//_ == nb_feats/_:
        nb_cols = nb_feats//_
    else:
        nb_cols = nb_feats//_+1

    fig, axs = plt.subplots(_, nb_cols, sharey=True)

    for i, _c in enumerate(final_list):
        ax = axs.flat[i]
        ax.hist(neighbors[[_c]], bins=20)
        ax.set_title(_c)
        fig.set_tight_layout(True)
    st.pyplot(fig)


def main():
    df, df_drop, cols, df_to_predict = get_train_test()
    id_client = id_client_side_bar()
    yes_no_feat_glob = yes_no_feat_glob_side_bar()
    nb_feats = nb_feats_side_bar()
    nb_neighbours = number_of_neighbours()
    options = multi_choice_neighbours()
    model = get_my_model()
    data_clients_std, data_clients_std_train = Calculate_all_scores(df_to_predict, df_drop, model)
    data_client_std, data_client = calculate_data_client(id_client, df_to_predict, data_clients_std)
    score = calculate_score_id_client(id_client, df_to_predict, data_client_std)
    score_to_score_str(score)
    df_feat_importance = features_importance_global(model, cols)

    if score != -1:

        if yes_no_feat_glob == 'Yes':
            plot_feat_importance_values(df_feat_importance)
        proba_client = predict_proba_client(data_client_std, model)
        plot_proba_client(proba_client)

        explainer = get_my_explainer()
        explanation_list = local_importance(model, data_client_std, explainer, nb_feats)
        final_list = find_loc_feat_importance(explanation_list, df_to_predict)
        hist_feats_loc(final_list, nb_feats, df_to_predict, data_client)

        if 'all clients mixed (1&0)' in options:
            neighbors = Calculate_neighbourhood(df, df_to_predict, nb_neighbours, final_list, data_client)
            plot_neigh(neighbors, final_list, nb_feats)
        """
        if 'Positive clients (1)' in options:
            neighbors_pos = Calculate_neighbourhood_positive(df, df_to_predict, nb_neighbours, final_list)
            plot_neigh(neighbors_pos, final_list, nb_feats)

        if 'Negatives clients (0)' in options:
            neighbors_neg = Calculate_neighbourhood_negative(df, df_to_predict, nb_neighbours, final_list)
            plot_neigh(neighbors_neg, final_list, nb_feats)
        """

# if __main__ == "main()":
main()
