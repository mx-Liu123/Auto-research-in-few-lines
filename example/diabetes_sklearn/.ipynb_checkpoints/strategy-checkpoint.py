import os
import math
import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Set deterministic behavior for reproducibility
torch.manual_seed(42)

class CnnLstmNetwork(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=64):
        super(CnnLstmNetwork, self).__init__()
        self.seq_len = seq_len
        
        # Layer 1: 1D CNN to extract local features
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2: LSTM
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.0) 
        
        # Heads - WTTE-RNN Architecture
        self.head_alpha = nn.Linear(hidden_dim, 1) # Predicts Scale (Alpha)
        self.head_beta = nn.Linear(hidden_dim, 1)  # Predicts Shape (Beta)
        self.head_clf = nn.Linear(hidden_dim, 1)   # Predicts Binary Classification

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)
        
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        
        feat = hn[-1]
        feat = self.dropout(feat)
        
        alpha = self.softplus(self.head_alpha(feat)) + 1e-6
        raw_beta = self.softplus(self.head_beta(feat)) + 1.01 
        beta = raw_beta
        
        clf_logits = self.head_clf(feat)
        
        return alpha, beta, clf_logits

def weibull_loss(alpha, beta, y_true):
    y_t = y_true + 1e-6
    term1 = torch.pow(y_t / alpha, beta)
    term2 = torch.log(beta)
    term3 = beta * torch.log(alpha)
    term4 = (beta - 1) * torch.log(y_t)
    nll = term1 - term2 + term3 - term4
    return torch.mean(nll)

class ModelStrategy:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
        # --- Strategy Configuration ---
        # Controlled by Strategy, requested by Evaluator
        self.required_train_days = int(os.getenv("REQUIRED_TRAIN_DAYS", "30"))
        
        self.seq_len = int(os.getenv("SEQ_LEN", "15"))
        self.train_sparsity = int(os.getenv("TRAIN_SPARSITY", "0")) 
        
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.initial_epochs = 300   
        self.incremental_epochs = 50 
        self.learning_rate = 0.001
        
        self.raw_columns = ['timestamp', 'ms_offset', 'open', 'high', 'low', 'close', 'volume', 'volume_stable']
        self.rsi_window = int(os.getenv("RSI_WINDOW", "14"))

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        close = X['close'].astype(float).values
        high = X['high'].astype(float).values
        low = X['low'].astype(float).values
        volume = X['volume'].astype(float).values

        # 1. Returns
        X['ret_1'] = X['close'].pct_change(1)
        X['ret_5'] = X['close'].pct_change(5)
        
        # 2. Relative SMA
        sma_10 = talib.SMA(close, timeperiod=10)
        X['dist_sma_10'] = (close - sma_10) / (sma_10 + 1e-6)
        
        # 3. Volatility
        atr = talib.ATR(high, low, close, timeperiod=14)
        X['atr_pct'] = atr / (close + 1e-6)
        
        # 4. RSI & ADX
        X['rsi'] = talib.RSI(close, timeperiod=self.rsi_window) / 100.0 
        X['adx'] = talib.ADX(high, low, close, timeperiod=14) / 100.0

        # 5. Relative Volume
        vol_sma = talib.SMA(volume, timeperiod=10)
        X['vol_rel'] = volume / (vol_sma + 1e-6)

        # 6. Momentum
        X['roc_5'] = talib.ROC(close, timeperiod=5)
        X['roc_10'] = talib.ROC(close, timeperiod=10)
        X['roc_5_sq'] = X['roc_5']**2
        X['roc_5_cub'] = X['roc_5']**3

        # 7. RPP
        X['highest_high'] = X['high'].rolling(window=self.seq_len).max()
        X['lowest_low'] = X['low'].rolling(window=self.seq_len).min()
        X['rpp'] = (X['close'] - X['lowest_low']) / (X['highest_high'] - X['lowest_low'] + 1e-6)
        X.drop(columns=['highest_high', 'lowest_low'], inplace=True) 

        # 9. Frequency
        X['close_low_freq'] = X['close'].rolling(window=20).mean()
        X['close_high_freq'] = X['close'] - X['close_low_freq']
        X['close_low_freq_rel'] = X['close_low_freq'] / (atr + 1e-6)
        X['close_high_freq_rel'] = X['close_high_freq'] / (atr + 1e-6)

        # 10. Causal Time Features
        time_diff_seconds = X['timestamp'].diff().dt.total_seconds()
        X['time_elapsed'] = time_diff_seconds.fillna(time_diff_seconds.mean()) / 60
        X['time_elapsed'] = np.maximum(0, X['time_elapsed'])
        X['log_time_elapsed'] = np.log1p(X['time_elapsed'])
        X['vol_adj_clock'] = np.log1p(X['time_elapsed'] * (X['atr_pct'] + 1e-6))
        X['vol_adj_clock'] = np.maximum(0, X['vol_adj_clock'])

        X['sin_hour'] = np.sin(2 * np.pi * X['timestamp'].dt.hour / 24.0)
        X['cos_hour'] = np.cos(2 * np.pi * X['timestamp'].dt.hour / 24.0)
        
        # 12. Time Since Rolling Extremes (Causal)
        time_since_max_high = X['high'].rolling(window=self.seq_len).apply(lambda x: x.argmax(), raw=False)
        X['time_since_max_high_norm'] = np.log1p(time_since_max_high / self.seq_len)

        time_since_min_low = X['low'].rolling(window=self.seq_len).apply(lambda x: x.argmin(), raw=False)
        X['time_since_min_low_norm'] = np.log1p(time_since_min_low / self.seq_len)

        # Robust fill
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X.drop(columns=['time_elapsed'], inplace=True, errors='ignore')

        feature_cols = [c for c in X.columns if c not in self.raw_columns]
        return X[feature_cols]

    def _create_sequences(self, X_scaled, y=None, y_clf=None):
        num_samples = len(X_scaled) - self.seq_len + 1
        if num_samples <= 0:
            return None, None, None
            
        X_seqs = []
        y_targets = []
        y_clf_targets = []
        
        for i in range(num_samples):
            X_seqs.append(X_scaled[i : i + self.seq_len])
            if y is not None:
                y_targets.append(y.iloc[i + self.seq_len - 1])
            if y_clf is not None:
                y_clf_targets.append(y_clf.iloc[i + self.seq_len - 1])
                
        X_tensor = torch.FloatTensor(np.array(X_seqs)).to(self.device)
        
        y_tensor = None
        if y is not None:
            y_tensor = torch.FloatTensor(np.array(y_targets)).reshape(-1, 1).to(self.device)
            
        y_clf_tensor = None
        if y_clf is not None:
            y_clf_tensor = torch.FloatTensor(np.array(y_clf_targets)).reshape(-1, 1).to(self.device)
            
        return X_tensor, y_tensor, y_clf_tensor

    def fit(self, df_train: pd.DataFrame, y_train: pd.Series):
        X_train = self.extract_features(df_train)
        valid_mask = ~X_train.isnull().any(axis=1) & ~y_train.isnull()
        X_clean = X_train[valid_mask]
        y_clean = y_train[valid_mask]
        
        # Sparsity Sampling
        if self.train_sparsity > 0 and len(y_clean) > 0:
            y_vals = y_clean.values
            diffs = np.diff(y_vals)
            event_indices = np.where(diffs > 0)[0] 
            all_indices = np.concatenate([[0], event_indices, [len(y_vals)-1]])
            
            keep_indices = []
            for i in range(len(all_indices) - 1):
                start_idx = all_indices[i]
                end_idx = all_indices[i+1]
                if end_idx > start_idx:
                    span = end_idx - start_idx
                    if span < self.train_sparsity:
                         indices = np.arange(start_idx, end_idx)
                    else:
                        target_end = start_idx + int(span * 0.8)
                        indices = np.linspace(start_idx, target_end, num=self.train_sparsity, dtype=int)
                        indices = np.unique(indices)
                        indices = indices[indices < len(y_vals)]
                    keep_indices.extend(indices)
            keep_indices = sorted(list(set(keep_indices)))
            X_clean = X_clean.iloc[keep_indices]
            y_clean = y_clean.iloc[keep_indices]

        y_clf_clean = (y_clean <= 60).astype(int)
        
        if len(X_clean) < self.seq_len + 10:
            return

        non_nan_y_clean = y_clean.dropna()
        if len(non_nan_y_clean) > 0:
            self.average_event_interval = non_nan_y_clean.mean()
        else:
            self.average_event_interval = 1.0 
        
        y_clean_normalized = y_clean / self.average_event_interval
        
        self.scaler.fit(X_clean) 
        X_scaled = self.scaler.transform(X_clean)
        
        X_tensor, y_tensor, y_clf_tensor = self._create_sequences(X_scaled, y_clean_normalized, y_clf_clean)
        if X_tensor is None: return

        input_dim = X_tensor.shape[2]
        
        if self.model is None:
            training_epochs = self.initial_epochs
            self.model = CnnLstmNetwork(input_dim, self.seq_len).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            training_epochs = self.incremental_epochs

        self.model.train()
        dataset = TensorDataset(X_tensor, y_tensor, y_clf_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        aux_task_weight = float(os.getenv("AUX_TASK_WEIGHT", "0.5"))
        beta_reg_weight = float(os.getenv("BETA_REGULARIZATION_WEIGHT", "0.0"))

        for epoch in range(training_epochs):
            for batch_X, batch_y, batch_y_clf in loader:
                self.optimizer.zero_grad()
                alpha, beta, clf_logits = self.model(batch_X)
                
                loss_nll = weibull_loss(alpha, beta, batch_y)
                loss_clf = F.binary_cross_entropy_with_logits(clf_logits, batch_y_clf.float())
                
                total_loss = loss_nll + aux_task_weight * loss_clf
                total_loss += beta_reg_weight * torch.mean(beta)
                
                total_loss.backward()
                self.optimizer.step()

    def predict(self, df_test: pd.DataFrame):
        if self.model is None: return np.zeros(len(df_test)), np.ones(len(df_test))

        self.model.eval()
        X_test = self.extract_features(df_test)
        X_test = X_test.ffill().fillna(0)
        X_scaled = self.scaler.transform(X_test)
        X_tensor, _, _ = self._create_sequences(X_scaled)
        
        if X_tensor is None: return np.zeros(len(df_test)), np.ones(len(df_test))

        with torch.no_grad():
            alpha_tensor, beta_tensor, _ = self.model(X_tensor) 
            
        alpha = alpha_tensor.cpu().numpy().flatten()
        beta = beta_tensor.cpu().numpy().flatten()
        
        predicted_time_norm = alpha * np.array([math.gamma(1 + 1/b) if b > 0 else 0 for b in beta])
        
        if hasattr(self, 'average_event_interval') and self.average_event_interval is not None:
            predicted_time = predicted_time_norm * self.average_event_interval
        else:
            predicted_time = predicted_time_norm
        
        predicted_time = np.maximum(0, predicted_time)
        pad_len = self.seq_len - 1
        mu_full = np.concatenate([np.full(pad_len, np.nan), predicted_time])
        sigma_full = np.concatenate([np.full(pad_len, np.nan), alpha * self.average_event_interval])
        
        return mu_full, sigma_full