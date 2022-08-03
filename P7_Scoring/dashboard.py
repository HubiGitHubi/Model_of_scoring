import math

import requests
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


# main function
def main():
    # URL = "http://192.168.1.27:8501/app/"
    # URL = "http://127.0.0.1:5000/app/"
    URL = "http://127.0.0.1:5000/app/"
    # Display the title
    st.title('Loan application scoring dashboard')

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

    # ___________ List of api Requests functions
    @st.cache
    def get_train_test_dashboard() -> object:
        # URL of the API + get_train_test
        api_url = URL + "get_train_test_values"

        # Requesting the API and saving the response
        response = requests.get(api_url)
        df = pd.DataFrame(st.json.loads(response.content['df']))
        df_drop = pd.DataFrame(st.json.loads(response.content['df_drop']))
        cols = pd.Series(st.json.loads(response.content['cols']))
        df_to_predict = pd.DataFrame(st.json.loads(response.content['df_to_predict']))
        return df, df_drop, cols, df_to_predict

    @st.cache
    def Calculate_all_data_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL + "Calculate_all_scores/"

        # Requesting the API and saving the response
        response = requests.get(api_url)

        data_clients_std = pd.DataFrame(st.json.loads(response.content['data_clients']))
        data_clients_std_train = pd.DataFrame(st.json.loads(response.content['data_clients_std_train']))
        return data_clients_std, data_clients_std_train

    @st.cache
    def calculate_data_client_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL + "calculate_data_client_values/"
        # Requesting the API and saving the response
        response = requests.get(api_url)
        data_client = pd.DataFrame(st.json.loads(response.content['data_client']))

        return data_client

    @st.cache
    def calculate_score_id_client_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL + "calculate_score_id_client_values/"
        # Requesting the API and saving the response
        response = requests.get(api_url)
        score = int(st.json.loads(response.content['score']))

        return score

    @st.cache
    def predict_proba_client_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL + "predict_proba_client_values/"
        # Requesting the API and saving the response
        response = requests.get(api_url)
        proba_client = pd.Series(st.json.loads(response.content['proba_client']))

        return proba_client

    @st.cache
    def features_importance_global_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL + "features_importance_global_values/"
        # Requesting the API and saving the response
        response = requests.get(api_url)
        df_feat_importance = pd.DataFrame(st.json.loads(response.content['df_feat_importance']))
        df_feat_importance = df_feat_importance.sort_values('feat_importance', ascending=False)

        return df_feat_importance

    @st.cache
    def local_importance_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL + "local_importance_values/"
        # Requesting the API and saving the response
        response = requests.get(api_url)
        explanation_list = pd.Series(st.json.loads(response.content['explanation_list']))
        explanation = pd.DataFrame(st.json.loads(response.content['explanation']))

        return explanation_list, explanation

    @st.cache
    def find_loc_feat_importance_dashboard():
        # URL of the API + Calculate_all_scores
        api_url = URL + "find_loc_feat_importance_values/"
        # Requesting the API and saving the response
        response = requests.get(api_url)
        final_list = pd.Series(st.json.loads(response.content['final_list']))

        return final_list

    def score_to_score_str(score):
        # markdown the status with color : green: accepted, red: refused, yellow : not in the db
        st.markdown("loan status :")
        if score == 0:
            st.success("accepted")
        elif score == 1:
            st.error("refused")
        else:
            st.warning("This client's not in the database")

    # ____________________ List of drawing functions
    def plot_proba_client(proba_client):
        # Plot the proba client
        st.write("Repayment rate")
        st.success(round(proba_client[0][0], 2))
        st.write("Default rate")
        st.error(round(proba_client[0][1], 2))

    def plot_feat_importance_values():
        # Plot the global features importance
        st.write("Global feature importance")
        fig = plt.figure(figsize=(15, 25))
        sns.barplot(data=df_feat_importance.reset_index(), x="feat_importance", y='Features')
        st.write(fig)

    def local_importance(explanation):
        with plt.style.context("ggplot"):
            st.pyplot(explanation.as_pyplot_figure())

    def hist_feats_loc(final_list, nb_feats, df_to_predict, id_client):
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
            fig.set_tight_layout(True)
        st.pyplot(fig)

    # Main program
    id_client = id_client_side_bar()
    yes_no_feat_glob = yes_no_feat_glob_side_bar()
    nb_feats = nb_feats_side_bar()
    df, df_drop, cols, df_to_predict = get_train_test_dashboard()
    data_clients_std, data_clients_std_train = Calculate_all_data_dashboard()
    data_client = calculate_data_client_dashboard()
    score = calculate_score_id_client_dashboard()
    proba_client = predict_proba_client_dashboard()
    if yes_no_feat_glob == 1:
        df_feat_importance = features_importance_global_dashboard()
    explanation_list, explanation = local_importance_dashboard()
    final_list = find_loc_feat_importance_dashboard()
    score_to_score_str(score)
    plot_proba_client(proba_client)
    local_importance(explanation)
    plot_feat_importance_values(df_feat_importance)
    hist_feats_loc(final_list, nb_feats, df_to_predict, id_client)


if __name__ == "__main__":
    main()
