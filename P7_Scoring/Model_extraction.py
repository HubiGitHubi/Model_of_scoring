import pickle

def get_my_model():
    # Charger le best model
    with open('best_model_CV', 'r', errors="ignore") as f1:
        f1.read()
    with open('best_model_CV', 'rb') as f1:
        my_model = pickle.load(f1)
    assert isinstance(my_model, object)
    return my_model
