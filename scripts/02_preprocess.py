import pandas as pd
import numpy as np
import holidays

def define_features(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    df['datedebut'] = pd.to_datetime(df['datedebut'], utc=True)
    df['hour'] = df['datedebut'].dt.hour
    df['day_of_week'] = df['datedebut'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5

    return df 

def add_lag_features(df):
    df = df.sort_values(['libelle', 'datedebut'])
    
    df['count_lag_1h'] = df.groupby('libelle')['comptage_1h'].shift(1)
    df['count_lag_24h'] = df.groupby('libelle')['comptage_1h'].shift(24)
    
    return df.dropna(subset=['count_lag_1h', 'count_lag_24h'])

def remove_outliers(df):
    df = df[df['comptage_1h'] >= 0]
    
    q_high = df['comptage_1h'].quantile(0.999)
    return df[df['comptage_1h'] < q_high]

def clean_data(data: pd.DataFrame, cols_to_drop: list, to_exclude: list) -> DataFrame:
    # Nettoyage
    df = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    df = df.dropna(subset=['comptage_1h'])
    df = df.drop(columns=[c for c in to_exclude if c in df.columns])
    # Encodage
    return pd.get_dummies(df, columns=['libelle'])

def add_cyclic_features(df):
    # hourly cycles (24h)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)


    # weekly cycles 
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # annual cycles 
    day_of_year = df['datedebut'].dt.dayofyear
    df['year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df

def add_context_features(df):
    fr_holidays = holidays.France()
    df['is_holiday'] = df['datedebut'].dt.date.apply(lambda x: x in fr_holidays)
    return df

if __name__ == "__main__":
    # Add features
    df_feat = define_features("./data/raw/filtered_counts.csv") # basic features : hour, day_of_week, is_weekend
    df_feat = add_context_features(df_feat) # Add is_holiday
    df_feat = add_lag_features(df_feat)
    df_feat = remove_outliers(df_feat)
    # Add cyclic features 
    processed_df_cyclic = add_cyclic_features(df_feat)

    # Drop useless columns 
    cols_to_drop = ['Geo Point', 'Geo Shape', 'gid', 'type', 'mdate', 'datefin']
    to_exclude = ['datedebut', 'ident', 'hour', 'day_of_week']
    processed_df = clean_data(processed_df_cyclic, cols_to_drop, to_exclude)

    processed_df.columns = [col.replace(' ', '_').replace('é', 'e').lower() for col in processed_df.columns]
    processed_df.to_csv("./data/processed/final_training_set.csv", index=False)
