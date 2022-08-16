import json
import math

import requests
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# main function
from sklearn.neighbors import NearestNeighbors


def main():
    #URL = "http://127.0.0.1:5000/"
    URL = "https://vast-falls-81708.herokuapp.com/app/"

    # Display the title
    st.title('Loan application scoring dashboard')

    def id_client_side_bar():
        # Add the page with settings and store the settings

        id_client = st.sidebar.number_input("Select the id client", 100001, 456250)
        return id_client

    def yes_no_feat_glob_side_bar():
        yes_no_feat_glob = st.sidebar.selectbox(
            "Plot the global features importance ? : ",
            ("Yes", "No"))
        return yes_no_feat_glob

    def yes_no_feat_local_side_bar():
        yes_no_feat_local = st.sidebar.selectbox(
            "Plot the local features importance ? : ",
            ("Yes", "No"))
        return yes_no_feat_local

    def nb_feats_side_bar():
        nb_feats = st.sidebar.slider(
            "How many local features do you want ?", 2, 15, step=1)
        return nb_feats

    def nb_neighbours():
        nb_neighbours = st.sidebar.slider(
            "How many local neighbours do you want ?", 10, 10000, step=20)
        return nb_neighbours

    def multi_choice_neighbours():
        options = st.sidebar.multiselect(
            'What kind of K neighbours graph do you want ?',
            ['Positive clients (1)', 'Negatives clients (0)', 'all clients mixed (1&0)'], ['all clients mixed (1&0)'])

        return options

    # ___________ List of api Requests functions
    def get_df_to_predict_dashboard() -> object:
        # URL of the API + get_df_predict_values
        api_url = URL+'get_df_predict_values/'
        # Requesting the API and saving the response
        response = requests.get(api_url)
        df_to_predict = pd.DataFrame(json.loads(response.content))

        return df_to_predict

    def get_df_dashboard() -> object:
        # URL of the API + get_df_values
        api_url = URL+'get_df_values/'
        # Requesting the API and saving the response
        response = requests.get(api_url)
        content = json.loads(response.content)
        df = pd.DataFrame(content)
        return df

    def calculate_data_client_dashboard():
        # URL of the API + calculate_data_client_values + id client
        api_url = URL+"calculate_data_client_values/?id_client="+str(id_client)
        # Requesting the API and saving the response
        response = requests.get(api_url)
        data_client = pd.DataFrame(json.loads(response.content))

        return data_client

    def calculate_score_id_client_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL+"calculate_score_id_client_values/?id_client="+str(id_client)
        # Requesting the API and saving the response
        response = requests.get(api_url)
        score = int(json.loads(response.content))

        return score

    def predict_proba_client_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL+"predict_proba_client_values/?id_client="+str(id_client)
        # Requesting the API and saving the response
        response = requests.get(api_url)
        proba_client = json.loads(response.content)

        return proba_client

    @st.cache
    def features_importance_global_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL+"features_importance_global_values/"
        # Requesting the API and saving the response
        response = requests.get(api_url)
        df_feat_importance = pd.DataFrame(json.loads(response.content))
        df_feat_importance = df_feat_importance.sort_values('feat_importance', ascending=False)

        return df_feat_importance

    def find_loc_feat_importance_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL+"find_loc_feat_importance_values/?id_client="+str(id_client)+"&nb_feats="+str(nb_feats)
        # Requesting the API and saving the response
        response = requests.get(api_url)
        final_list = pd.Series(json.loads(response.content))

        return final_list

    def score_to_score_str(score):
        # markdown the status with color : green: accepted, red: refused, yellow : not in the db
        st.markdown("loan status :")
        if score == 0:
            st.success("accepted")
        elif score == 1:
            st.error("refused")

    # ____________________ List of drawing functions

    def plot_proba_client(proba_client):
        # Plot the proba client
        st.write("Repayment rate")
        st.success(round(proba_client[0], 2))
        st.write("Default rate")
        st.error(round(proba_client[1], 2))

    def plot_feat_importance_values(df_feat_importance):
        # Plot the global features importance
        st.write("Global feature importance")
        fig = plt.figure(figsize=(15, 25))
        df_feat_importance_abs = abs(df_feat_importance).sort_values(by='feat_importance', ascending=False).iloc[0:15]
        sns.barplot(data=df_feat_importance_abs.reset_index(), x="feat_importance", y='index')
        st.write(fig)

    def hist_feats_loc(final_list, nb_feats, df_to_predict, data_client):
        # Plot the number of chosen local most important feats
        st.write("Locals feature importance")
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

            ax.set_title(_c)
            fig.set_tight_layout(True)
        st.pyplot(fig)

    # find nearest neighbors among the training set

    def Calculate_neighbourhood(df, nb_neighbours, final_list, data_client):
        df_all = df
        # return the closest neighbors final feats list (nb_neighbours chosen by the user)
        neighbors = NearestNeighbors(n_neighbors=nb_neighbours).fit(df_all.drop(['SK_ID_CURR', 'TARGET'], axis=1))

        index_neighbors = list(neighbors.kneighbors(X=data_client.drop(['SK_ID_CURR', 'score'], axis=1).values,
                                                    n_neighbors=nb_neighbours, return_distance=False).ravel())

        neighbors = []
        df_all.index = df_all.SK_ID_CURR
        for i in index_neighbors:
            neighbors.append(df_all.iloc[i, :])
        neighbors = pd.DataFrame(neighbors, columns=df_all.columns)
        neighbors = neighbors.loc[:, final_list]
        return neighbors

    def Calculate_neighbourhood_positive(df, nb_neighbours, final_list, data_client):
        df_pos = df[df["TARGET"] == 1]

        # return the closest neighbors final feats list (nb_neighbours chosen by the user)
        neighbors_pos = NearestNeighbors(n_neighbors=nb_neighbours).fit(df_pos.drop(['SK_ID_CURR', 'TARGET'], axis=1))

        index_neighbors = list(neighbors_pos.kneighbors(X=data_client.drop(['SK_ID_CURR', 'score'], axis=1),
                                                        n_neighbors=nb_neighbours, return_distance=False).ravel())
        neighbors_pos = []
        df_pos.index = df_pos.SK_ID_CURR
        for i in index_neighbors:
            neighbors_pos.append(df_pos.iloc[i, :])
        neighbors_pos = pd.DataFrame(neighbors_pos, columns=df_pos.columns)
        neighbors_pos = neighbors_pos.loc[:, final_list]

        return neighbors_pos

    def Calculate_neighbourhood_negative(df, nb_neighbours, final_list, data_client):
        df_neg = df[df["TARGET"] == 0]

        # return the closest neighbors final feats list (nb_neighbours chosen by the user)
        neighbors_neg = NearestNeighbors(n_neighbors=nb_neighbours).fit(df_neg.drop(['SK_ID_CURR', 'TARGET'], axis=1))

        index_neighbors = list(neighbors_neg.kneighbors(X=data_client.drop(['SK_ID_CURR', 'score'], axis=1),
                                                        n_neighbors=nb_neighbours, return_distance=False).ravel())
        neighbors_neg = []
        df_neg.index = df_neg.SK_ID_CURR
        for i in index_neighbors:
            neighbors_neg.append(df_neg.iloc[i, :])
        neighbors_neg = pd.DataFrame(neighbors_neg, columns=df_neg.columns)
        neighbors_neg = neighbors_neg.loc[:, final_list]

        return neighbors_neg

    def plot_neigh(neighbors, final_list, nb_feats, data_client):
        # Plot local most important feats for the number of chosen neighbours

        _ = math.ceil(math.sqrt(len(final_list)))
        if nb_feats//_ == nb_feats/_:
            nb_cols = nb_feats//_
        else:
            nb_cols = nb_feats//_+1

        fig, axs = plt.subplots(_, nb_cols, sharey=True)

        for i, _c in enumerate(final_list):
            ax = axs.flat[i]
            ax.axvline(data_client[_c][0], color='red')
            ax.hist(neighbors[[_c]], bins=20)
            ax.set_title(_c)
            fig.set_tight_layout(True)
        st.pyplot(fig)

    # Main program
    # Return values from button and settings page
    id_client = id_client_side_bar()
    yes_no_feat_glob = yes_no_feat_glob_side_bar()
    yes_no_feat_local = yes_no_feat_local_side_bar()
    nb_feats = nb_feats_side_bar()
    options = multi_choice_neighbours()
    nb_neighbours = nb_neighbours()

    # Import datas from Flask
    df = get_df_dashboard()

    df_to_predict = get_df_to_predict_dashboard()
    data_client = calculate_data_client_dashboard()
    score = calculate_score_id_client_dashboard()

    # Plot the dashboard

    if score != -1:

        proba_client = predict_proba_client_dashboard()
        score_to_score_str(score)
        plot_proba_client(proba_client)

        if yes_no_feat_glob == 'Yes':
            df_feat_importance = features_importance_global_dashboard()
            plot_feat_importance_values(df_feat_importance)

        final_list = find_loc_feat_importance_dashboard()
        if yes_no_feat_local == 'Yes':
            hist_feats_loc(final_list, nb_feats, df_to_predict, data_client)

        if 'all clients mixed (1&0)' in options:
            st.write("all clients mixed (1&0) neighbours")
            neighbors = Calculate_neighbourhood(df, nb_neighbours, final_list, data_client)
            plot_neigh(neighbors, final_list, nb_feats, data_client)

        if 'Positive clients (1)' in options:
            st.write("Positive clients (1) neighbours")
            neighbors_pos = Calculate_neighbourhood_positive(df, nb_neighbours, final_list, data_client)
            plot_neigh(neighbors_pos, final_list, nb_feats, data_client)

        if 'Negatives clients (0)' in options:
            st.write("Negatives clients (0) neighbours")
            neighbors_neg = Calculate_neighbourhood_negative(df, nb_neighbours, final_list, data_client)
            plot_neigh(neighbors_neg, final_list, nb_feats, data_client)
    else:
        st.warning("This client's not in the database")


if __name__ == "__main__":
    main()
