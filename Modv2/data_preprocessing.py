import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

# Configuration
data_dir = 'data'          # folder with your collected CSVs
public_dir = 'public_data' # folder with Balabit CSVs (optional)
seq_len = 50               # sequence length for LSTM / diffusion
output_prefix = 'public_'         # change to 'public_' when running on Balabit

# Function to read and standardize one CSV
def read_and_standardize_csv(filepath):
    df = pd.read_csv(filepath)
    
    # Handle Balabit format
    if 'CLIENT_TIMESTAMP' in df.columns:
        logging.info(f"Detected Balabit format in {filepath}")
        # Use CLIENT_TIMESTAMP (local client time) as it's more consistent for movement
        if 'X' in df.columns and 'Y' in df.columns:
            df = df[['CLIENT_TIMESTAMP', 'X', 'Y']].copy()
            df.columns = ['timestamp', 'x', 'y']
        else:
            raise ValueError(f"Balabit file {filepath} missing X or Y columns")
        
        # Optional: keep only Move events (cleaner for pure trajectories)
        # if 'STATE' in df.columns:
        #     df = df[df['STATE'] == 'Move']
        
        # Make timestamps relative (start from 0)
        df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    
    # Handle your own collected format (timestamp, x, y)
    elif 'timestamp' in df.columns and 'x' in df.columns and 'y' in df.columns:
        logging.info(f"Detected custom format in {filepath}")
        df = df[['timestamp', 'x', 'y']].copy()
    else:
        raise ValueError(f"Unknown CSV format in {filepath}. Expected timestamp/x/y or Balabit columns.")
    
    # Ensure numeric types
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna()
    
    return df

# 1. Load all CSVs from the chosen directory
dfs = []
target_dir = data_dir  # change to public_dir when testing Balabit
for file in os.listdir(target_dir):
    if file.lower().endswith('.csv'):
        path = os.path.join(target_dir, file)
        try:
            df = read_and_standardize_csv(path)
            if len(df) > 100:  # skip very tiny files
                dfs.append(df)
                logging.info(f"Loaded {file} — {len(df)} rows")
            else:
                logging.warning(f"Skipped tiny file {file}")
        except Exception as e:
            logging.error(f"Failed to process {file}: {e}")

if not dfs:
    raise ValueError("No valid CSV files found in the directory.")

full_df = pd.concat(dfs, ignore_index=True)
logging.info(f"Merged total points: {len(full_df):,}")

# 2. Compute derived features (exactly as in your PDFs)
full_df['dx'] = full_df['x'].diff().fillna(0)
full_df['dy'] = full_df['y'].diff().fillna(0)
full_df['dt'] = full_df['timestamp'].diff().fillna(0.008)  # typical ~120 Hz fallback
full_df['speed'] = np.sqrt(full_df['dx']**2 + full_df['dy']**2) / full_df['dt'].clip(lower=1e-6)

# Drop any remaining NaNs (should be minimal)
full_df = full_df.dropna().reset_index(drop=True)

# 3. Normalize features (important for stable training)
features = ['dx', 'dy', 'speed', 'dt']
scaler = MinMaxScaler()
full_df[features] = scaler.fit_transform(full_df[features])

# Optional: save normalized raw data
full_df.to_pickle(f'{output_prefix}processed_data.pkl')
logging.info("Saved processed_data.pkl")

# 4. Create sliding-window sequences (LSTM input → predict next dx,dy)
X, y = [], []
for i in range(len(full_df) - seq_len):
    seq = full_df[features].iloc[i:i+seq_len].values
    next_val = full_df[['dx', 'dy']].iloc[i+seq_len].values
    X.append(seq)
    y.append(next_val)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

logging.info(f"Created {len(X)} sequences (seq_len={seq_len})")

# 5. Train / Val / Test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Save splits
np.savez(
    f'{output_prefix}sequences.npz',
    X_train=X_train, y_train=y_train,
    X_val=X_val,   y_val=y_val,
    X_test=X_test, y_test=y_test
)
logging.info(f"Saved sequences.npz → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")