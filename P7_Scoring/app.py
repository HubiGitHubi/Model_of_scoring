# import _json
#Affichier uniquement le nbre de glob feat max ou la quantité demandé dans le slider
from flask import jsonify, Flask, request
from P7_Scoring.Streamlit_Fonctions import *

cd  C:/Users/33646/Documents/OpenClassroom/'Projet 7'/Model_of_scoring/P7_Scoring
streamlit run app.py

# Local URL: http: // localhost: 8501
# Network URL: http: // 192.168.1.27:8501
#http://192.168.1.22:8501


# Import data, model, explainer
df, df_drop, cols, df_to_predict = get_train_test()
model = get_my_model()
data_clients_std, data_clients_std_train = Calculate_all_scores(df_to_predict, df_drop, model)
df_feat_importance = features_importance_global(model, cols)
explainer = get_my_explainer()

app = Flask(__name__)

@app.route("/")
def beginning():
    return "Model and data are now loaded"


@app.route("/app/get_my_model_values")
def get_my_model(my_model):
    my_model_json = st.json.loads(my_model.to_json())
    return jsonify({'my_model': my_model_json})


@app.route("/app/get_my_explainer_values")
def get_my_explainer(explainer):
    # Charge the explainer'

    explainer_json = st.json.loads(explainer.to_json())

    return jsonify({'explainer': explainer_json})


@app.route("/app/get_train_test_values")
def get_train_test(df,df_drop,cols,df_to_predict) -> object:
    df_json = st.json.loads(df.to_json())
    df_drop_json = st.json.loads(df_drop.to_json())
    cols_json = st.json.loads(cols.to_json())
    df_to_predict_json = st.json.loads(df_to_predict.to_json())

    return jsonify({'df': df_json,
                    'df_drop': df_drop_json,
                    'cols': cols_json,
                    'df_to_predict': df_to_predict_json})


@app.route("/app/Calculate_all_scores_values")
def Calculate_all_scores(data_clients_std, data_clients_std_train):
    data_clients_std_json = st.json.loads(data_clients_std.to_json())
    data_clients_std_train_json = st.json.loads(data_clients_std_train.to_json())
    return jsonify({'data_clients_std': data_clients_std_json,
                    'data_clients_std_train': data_clients_std_train_json})


@app.route("/app/calculate_data_client_values")
def calculate_data_client(df_to_predict, data_clients_std):
    # Return the data of the chosen client
    id_client = int(request.args.get('SK_ID_CURR'))
    data_client = data_clients_std[df_to_predict.SK_ID_CURR == id_client]
    data_client_json = st.json.loads(data_client.to_json())

    return jsonify({'data_client': data_client_json})


@app.route("/app/calculate_score_id_client_values")
def calculate_score_id_client(df_to_predict, data_client):
    # Return the score of the chosen client. If the client is not in the dtb, return -1
    id_client = int(request.args.get('SK_ID_CURR'))

    if len(data_client) > 0:
        score = int(df_to_predict.score[df_to_predict.SK_ID_CURR == id_client])
    else:
        score = -1
    score_json = st.json.loads(score.to_json())

    return jsonify({'score': score_json})


@app.route("/app/predict_proba_client_values")
def predict_proba_client(data_client, model):
    # Return proba of success/failure of a client
    proba_client = model.predict_proba(data_client)
    proba_client_json = st.json.loads(proba_client.to_json())

    return jsonify({'proba_client': proba_client_json})


@app.route("/app/features_importance_global_values")
def features_importance_global(df_feat_importance):
    # Calculate the global features importance
    df_feat_importance_json = st.json.loads(df_feat_importance.to_json())

    return jsonify({'df_feat_importance': df_feat_importance_json})


@app.route("/app/local_importance_values")
def local_importance(model, data_client, explainer, nb_feats):
    explanation = explainer.explain_instance(data_client.values.reshape(-1),
                                             model.predict_proba,
                                             num_features=nb_feats)
    explanation_list = explanation.as_list()
    explanation_list_json = st.json.loads(explanation_list.to_json())
    explanation_json = st.json.loads(explanation_list.to_json())
    return jsonify({'explanation_list': explanation_list_json,
                   'Explanation': explanation_json})


@app.route("/app/find_loc_feat_importance_values")
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
    final_list_json = st.json.loads(final_list.to_json())

    return jsonify({'final_list': final_list_json})


if __name__ == main():
    app.run()
