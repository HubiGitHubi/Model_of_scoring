import pandas as pd


def get_my_df() -> object:
    path = 'C:/Users/33646/Documents/OpenClassroom/Projet 7/Model_of_scoring/Datas/data_clients.csv'
    path = 'Datas/data_clients.csv'
    df = pd.read_csv(path)
    df_drop = df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    cols = pd.DataFrame(df_drop.columns, columns=['Features'])

    return df, df_drop, cols



