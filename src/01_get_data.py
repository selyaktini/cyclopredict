import pandas as pd


"""Function to laod and filter data from a csv file depending on a list of targets""" 
def load_and_filter_data(input_path: str, targets: list) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=';')
    return df[df["libelle"].isin(targets)].copy()

"""Function to save selected data (by load_and_filter_data) in file output_path"""
def save_raw_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    TARGET_STATIONS = [
        'Totem vélo Sens Entrant', 'Totem vélo Sens Sortant',
        'Bordeaux vers Haut Leveque', 'Haut Leveque vers Bordeaux'
    ]
    raw_df = load_and_filter_data("./data/raw/bordeaux_velo_historique.csv", TARGET_STATIONS)
    save_raw_data(raw_df, "./data/raw/filtered_counts.csv")
