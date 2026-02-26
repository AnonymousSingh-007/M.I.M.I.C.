import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

# Merge all CSVs in data/
data_dir = 'data'
dfs = []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)
logging.info(f"Merged {len(dfs)} files, total points: {len(full_df)}")

# Compute features
full_df['dx'] = full_df['x'].diff().fillna(0)
full_df['dy'] = full_df['y'].diff().fillna(0)
full_df['dt'] = full_df['timestamp'].diff().fillna(0.001)  # Avoid zero dt
full_df['speed'] = np.sqrt(full_df['dx']**2 + full_df['dy']**2) / full_df['dt']
full_df = full_df.dropna()  # Drop first row

# Normalize (x,y to [0,1] for screen, dx/dy/speed/dt min-max)
scaler = MinMaxScaler()
features = ['dx', 'dy', 'speed', 'dt']
full_df[features] = scaler.fit_transform(full_df[features])
full_df.to_pickle('processed_data.pkl')  # Save scaled
logging.info("Features computed and normalized")

# Create sequences for LSTM/Diffusion (seq_len=50, predict next)
seq_len = 50
X, y = [], []
for i in range(len(full_df) - seq_len):
    seq = full_df[features].iloc[i:i+seq_len].values
    next_val = full_df[['dx', 'dy']].iloc[i+seq_len].values  # Predict next deltas
    X.append(seq)
    y.append(next_val)
X = np.array(X)
y = np.array(y)

# Split: 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
np.savez('sequences.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
logging.info(f"Sequences created: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")