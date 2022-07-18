import pickle
#import imblearn *

def get_my_model() -> object:
    """

    :rtype: object
    """
    # Charger le best model
    # with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/best_model', 'rb') as f1:
    # my_model = pickle.load(f1)
    # return my_model

    try:
        with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/best_model', 'rb') as f1:
            my_model = pickle.load(f1)
    except:
        with open('Datas/best_model', 'rb') as f1:
            my_model = pickle.load(f1)
    return my_model


def get_my_explainer(data_clients_std, cols):
    # Charger l'explainer'

    # with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/explainer', 'rb') as f:
    # explainer = dill.load(f, errors="ignore")

    # return explainer

    try:
        with open('C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    except:
        with open('Datas/explainer', 'rb') as f:
            explainer = pickle.load(f, errors="ignore")
    return explainer
