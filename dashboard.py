import json
import math
import requests
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def main():
    #URL = "http://127.0.0.1:5000/"  #Local test
    URL = "https://happyhappyapy.herokuapp.com/"
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

    st.cache()

    def get_df_to_predict_dashboard() -> object:
        # URL of the API + get_df_predict_values
        api_url = URL+'get_df_predict_values/'
        # Requesting the API and saving the response
        response = requests.get(api_url)
        df_to_predict = pd.DataFrame(json.loads(response.content))

        return df_to_predict

    st.cache()

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

    st.cache()

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

    def find_loc_feat_importance_dashboard_as_list():
        # URL of the API + Calculate_all_scores
        api_url = URL+"find_loc_feat_importance_values_as_list/?id_client="+str(id_client)+"&nb_feats="+str(nb_feats)
        # Requesting the API and saving the response
        response = requests.get(api_url)
        expl = pd.DataFrame(json.loads(response.content), columns=['Features', 'Feature_importance'])
        fig = plt.figure(figsize=(15, round(len(expl)/1.5)))

        sns.barplot(data=expl
                    .sort_values(by='Feature_importance', ascending=False), x='Feature_importance', y='Features')
        plt.title("Local features importance")
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        st.write(fig)



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
        st.success(str(round(proba_client[0], 2)*100)+'%')
        st.write("Default rate")
        st.error(str(round(proba_client[1], 2)*100)+'%')

    def plot_feat_importance_values(df_feat_importance):
        # Plot the global features importance
        list_feat_importance = abs(df_feat_importance).sort_values(by='feat_importance', ascending=False).index[0:15]
        fig = plt.figure(figsize=(13, 10))

        sns.barplot(data=df_feat_importance.loc[list_feat_importance, :].reset_index()
                    .sort_values(by='feat_importance', ascending=False), x="feat_importance", y='index')
        plt.title("Global features importance")
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
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
        df_all = df[final_list]
        df_all_std = pd.DataFrame(StandardScaler().fit_transform(df_all), columns=df_all.columns)

        data_client_std = pd.DataFrame(StandardScaler().fit(df_all_std).transform(
            data_client[final_list]), columns=final_list)

        # return the closest neighbors final feats list (nb_neighbours chosen by the user)
        if nb_neighbours > len(df_all):
            st.write('(max positives neighbours :', len(df_all), ')')
            nb_neighbours = len(df_all)

        neighbors = NearestNeighbors(n_neighbors=nb_neighbours).fit(df_all_std)

        index_neighbors = list(neighbors.kneighbors(X=data_client_std,
                                                    n_neighbors=nb_neighbours, return_distance=False).ravel())

        neighbors = []
        df_all.index = df.SK_ID_CURR
        for i in index_neighbors:
            neighbors.append(df_all.iloc[i, :])
        neighbors = pd.DataFrame(neighbors, columns=final_list)
        return neighbors

    def Calculate_neighbourhood_positive(df, nb_neighbours, final_list, data_client):
        df_pos_ext = df[df["TARGET"] == 1]
        df_pos = df_pos_ext[final_list]
        df_pos_std = pd.DataFrame(StandardScaler().fit_transform(df_pos), columns=df_pos.columns)

        data_client_std = pd.DataFrame(StandardScaler().fit(df_pos_std).transform(
            data_client[final_list]), columns=final_list)

        # return the closest neighbors final feats list (nb_neighbours chosen by the user)
        if nb_neighbours > len(df_pos):
            st.write('(max positives neighbours :', len(df_pos), ')')
            nb_neighbours = len(df_pos)

        neighbors_pos = NearestNeighbors(n_neighbors=nb_neighbours).fit(df_pos_std)

        index_neighbors = list(neighbors_pos.kneighbors(X=data_client_std,
                                                        n_neighbors=nb_neighbours, return_distance=False).ravel())
        neighbors_pos = []
        df_pos.index = df_pos_ext.SK_ID_CURR
        for i in index_neighbors:
            neighbors_pos.append(df_pos.iloc[i, :])
        neighbors_pos = pd.DataFrame(neighbors_pos, columns=df_pos.columns)

        return neighbors_pos

    def Calculate_neighbourhood_negative(df, nb_neighbours, final_list, data_client):
        df_neg_ext = df[df["TARGET"] == 0]
        df_neg = df_neg_ext[final_list]
        df_neg_std = pd.DataFrame(StandardScaler().fit_transform(df_neg), columns=df_neg.columns)

        data_client_std = pd.DataFrame(StandardScaler().fit(df_neg_std).transform(
            data_client[final_list]), columns=final_list)

        # return the closest neighbors final feats list (nb_neighbours chosen by the user)
        if nb_neighbours > len(df_neg):
            st.write('(max negatives neighbours :', len(df_neg), ')')
            nb_neighbours = len(df_neg)

        neighbors_neg = NearestNeighbors(n_neighbors=nb_neighbours).fit(df_neg_std)

        index_neighbors = list(neighbors_neg.kneighbors(X=data_client_std,
                                                        n_neighbors=nb_neighbours, return_distance=False).ravel())
        neighbors_neg = []
        df_neg.index = df_neg_ext.SK_ID_CURR
        for i in index_neighbors:
            neighbors_neg.append(df_neg.iloc[i, :])
        neighbors_neg = pd.DataFrame(neighbors_neg, columns=df_neg.columns)
        neighbors_neg = neighbors_neg.loc[:, final_list]

        return neighbors_neg

    def plot_neigh(neighbors, final_list, nb_feats, data_client):
        # Plot local most important feats for the number of chosen neighbours

        nb_cols = 2

        if nb_feats//nb_cols == nb_feats/nb_cols:
            nb_lignes = nb_feats//nb_cols+1
        else:
            nb_lignes = nb_feats//nb_cols

        fig, axs = plt.subplots(nb_lignes, nb_cols, sharey=True)

        for i, _c in enumerate(final_list):
            ax = axs.flat[i]
            ax.axvline(data_client[_c][0], color='red')
            ax.hist(neighbors[[_c]])
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
            # hist_feats_loc(final_list, nb_feats, df_to_predict, data_client)
            find_loc_feat_importance_dashboard_as_list()

        if 'all clients mixed (1&0)' in options:
            st.write("all clients mixed (1&0) neighbours :")
            neighbors = Calculate_neighbourhood(df, nb_neighbours, final_list, data_client)
            plot_neigh(neighbors, final_list, nb_feats, data_client)

        if 'Positive clients (1)' in options:
            st.write("Refused clients (1) neighbours :")
            neighbors_pos = Calculate_neighbourhood_positive(df, nb_neighbours, final_list, data_client)
            plot_neigh(neighbors_pos, final_list, nb_feats, data_client)

        if 'Negatives clients (0)' in options:
            st.write("Authorised clients (0) neighbours :")
            neighbors_neg = Calculate_neighbourhood_negative(df, nb_neighbours, final_list, data_client)
            plot_neigh(neighbors_neg, final_list, nb_feats, data_client)
    else:
        st.warning("This client's not in the database, here is the list of clients")
        st.write(df_to_predict.SK_ID_CURR)


if __name__ == "__main__":
    main()
