import pickle
import dill
from lime import lime_tabular


def get_my_model() -> object:
    """

    :rtype: object
    """
    # Charger le best model
    #with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/best_model', 'r',
              #errors="ignore") as f1:
        #f1.read()
    with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/best_model', 'rb') as f1:
        my_model = pickle.load(f1)
    return my_model


def get_my_explainer(data_clients_std, cols):
    # Charger l'explainer'

    with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/explainer', 'rb') as f:
        explainer = dill.load(f, errors="ignore")


    #explainer = lime_tabular.LimeTabularExplainer(data_clients_std,
    #                                             mode="classification",
    #                                             class_names=[0, 1],
    #                                              feature_names=cols)

    return explainer
