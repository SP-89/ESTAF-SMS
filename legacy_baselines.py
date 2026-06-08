import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats
import warnings
from sklearn.feature_selection import SelectKBest, f_regression

# Warning，
warnings.filterwarnings('ignore')

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")


# Set random seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ==========================================
# 1. Data Loading (Data Loading)
# ==========================================
def load_and_preprocess_data(data_dir, excel_path):
    """
    Load EEG data with preprocessing
    """
    print(f"📂 Loading from {data_dir} loading data...")

    if not os.path.exists(data_dir) or not os.path.exists(excel_path):
        print("❌ Data path not found。")
        return None, None, None, None

    # Read Excel table
    df = pd.read_excel(excel_path)
    target_columns = ['ADL', 'FMA', 'upper_FMA']

    # Fill missing values
    for col in target_columns:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    all_features = []
    all_targets = []
    valid_indices = []

    # h5File
    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        h5_file = os.path.join(data_dir, f"{patient_id}.h5")

        if not os.path.exists(h5_file):
            continue

        try:
            with h5py.File(h5_file, 'r') as f:
                if 'psd_features' in f:
                    psd_features = f['psd_features'][:]  # (100, 29, 90)

                    # Reshape and normalize
                    time_features = psd_features.reshape(100, 29 * 90)
                    scaler = StandardScaler()
                    time_features_scaled = scaler.fit_transform(time_features)

                    all_features.append(time_features_scaled)
                    targets = [row[col] for col in target_columns]
                    all_targets.append(targets)
                    valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading {h5_file}: {e}")
            continue

    if not all_features:
        print("❌ No valid data loaded。")
        return None, None, None, None

    X = np.array(all_features)  # (N, 100, 2610)
    y = np.array(all_targets)  # (N, 3)

    # ： (N, 100, 10)
    print("⚡  (SelectKBest)...")
    n_samples, n_times, n_feats = X.shape
    X_reshaped = X.reshape(n_samples * n_times, n_feats)
    # targety
    y_reshaped = np.repeat(np.mean(y, axis=1), n_times)

    selector = SelectKBest(f_regression, k=10)  # 10
    X_selected = selector.fit_transform(X_reshaped, y_reshaped)
    X_final = X_selected.reshape(n_samples, n_times, 10)

    print(f"✅ Data LoadingDone: X shape={X_final.shape}, y shape={y.shape}")
    return X_final, y, df, valid_indices


class EEGDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==========================================
# 2.  (Model Definitions)
# ==========================================

# --- 2.1  ---
class SklearnBaseline:
    def __init__(self, model_type='svr'):
        if model_type == 'svr':
            self.model = SVR(kernel='rbf', C=10, epsilon=0.1)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    def fit(self, X_train, y_train):

        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_flat, y_train)

    def predict(self, X_test):
        X_flat = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_flat)


# --- 2.2  (Standard CNN & LSTM) ---
class StandardCNN(nn.Module):
    def __init__(self, input_size, seq_len):
        super(StandardCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        # MaxPool1d(2)，
        self.fc = nn.Linear(32 * (seq_len // 2), 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.fc(x).squeeze()


class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()


# --- 2.3 Proposed Model (Full) ---
class ProposedModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(ProposedModel, self).__init__()
        # 1. Local Feature Extraction (CNN)
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 2. Global Temporal Modeling (LSTM)
        self.lstm = nn.LSTM(16, hidden_size, batch_first=True)

        # 3. Stream 1: Temporal Attention
        self.attn_linear = nn.Linear(hidden_size, 1)

        # 5. Fusion & Regression
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        attn_scores = torch.tanh(self.attn_linear(lstm_out))
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        global_avg = torch.mean(lstm_out, dim=1)
        global_max, _ = torch.max(lstm_out, dim=1)

        combined = torch.cat((context_vector, global_avg, global_max), dim=1)
        output = self.fc(combined)
        return output.squeeze()


# --- 2.4 Ablation Variants ---
class AblationNoAttention(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(AblationNoAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        global_avg = torch.mean(lstm_out, dim=1)
        global_max, _ = torch.max(lstm_out, dim=1)
        combined = torch.cat((global_avg, global_max), dim=1)
        return self.fc(combined).squeeze()


class AblationStream1Only(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(AblationStream1Only, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(16, hidden_size, batch_first=True)
        self.attn_linear = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(torch.tanh(self.attn_linear(lstm_out)), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context).squeeze()


# ==========================================
# 3.  (Utilities)
# ==========================================
def train_and_evaluate(model_name, model_class, X, y, input_size, seq_len, k_folds=5, epochs=100):
    # K
    y_stratify = np.digitize(y, np.percentile(y, [33, 66]))
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = {'r': [], 'mae': []}

    for train_idx, val_idx in kf.split(X, y_stratify):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]


        if model_name in ['SVR', 'Random Forest']:
            model = model_class(model_type='svr' if model_name == 'SVR' else 'rf')
            model.fit(X_train, y_train)
            preds = model.predict(X_val)


        else:
            if model_name == 'Standard CNN':
                model = model_class(input_size, seq_len).to(device)
            else:
                model = model_class(input_size).to(device)

            train_dataset = EEGDataset(X_train, y_train)
            val_dataset = EEGDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

            optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

            best_val_loss = float('inf')
            best_model_state = None

            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()


                model.eval()
                val_loss = 0
                with torch.no_grad():
                    val_x = torch.FloatTensor(X_val).to(device)
                    val_y_tensor = torch.FloatTensor(y_val).to(device)
                    val_preds_tensor = model(val_x)
                    val_loss = criterion(val_preds_tensor, val_y_tensor).item()

                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()


            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(X_val).to(device)
                preds = model(val_x).cpu().numpy()


        r, _ = stats.pearsonr(y_val, preds)
        mae = mean_absolute_error(y_val, preds)

        metrics['r'].append(r)
        metrics['mae'].append(mae)

    return {k: np.mean(v) for k, v in metrics.items()}, {k: np.std(v) for k, v in metrics.items()}


# ==========================================
# 4.  (Main Execution)
# ==========================================
if __name__ == "__main__":
    # Configure paths ()
    data_directory = "h5_files"
    excel_file_path = "metadata.xlsx"

    # 1. 
    X, y_all, _, _ = load_and_preprocess_data(data_directory, excel_file_path)

    if X is None:
        print("Program：loading data。")
        exit(1)

    #  sequence length
    # X.shape = (n_samples, seq_len, feat_dim)
    SEQ_LEN = X.shape[1]
    FEAT_DIM = X.shape[2]
    EPOCHS = 150

    print(f"📊 : samples={X.shape[0]}, ={SEQ_LEN}, ={FEAT_DIM}")

    # y_all  (N, 3),  [ADL, FMA, upper_FMA]
    # : Upper FMA (2), Total FMA (1), ADL (0)
    tasks = ['Upper FMA', 'Total FMA', 'ADL']
    task_indices = [2, 1, 0]

    print("\n" + "=" * 60)
    print("🧪 Experiment 1: ")
    print("=" * 60)

    models_config = [
        ('SVR', SklearnBaseline),
        ('Random Forest', SklearnBaseline),
        ('Standard CNN', StandardCNN),
        ('Standard LSTM', StandardLSTM),
        ('Proposed', ProposedModel)
    ]

    final_comparison_table = {}

    for task_name, task_idx in zip(tasks, task_indices):
        print(f"\nEvaluating task: {task_name} ...")
        y_task = y_all[:, task_idx]

        task_results = {}

        for model_name, model_class in models_config:
            mean_metrics, std_metrics = train_and_evaluate(
                model_name, model_class, X, y_task, FEAT_DIM, SEQ_LEN, epochs=EPOCHS
            )

            task_results[model_name] = {'r': mean_metrics['r'], 'mae': mean_metrics['mae']}
            print(f"  -> {model_name}: r={mean_metrics['r']:.4f}, MAE={mean_metrics['mae']:.2f}")

        final_comparison_table[task_name] = task_results

    print("\n" + "=" * 60)
    print("🧪 Experiment 2: Experiment (Ablation Study) -  Upper FMA")
    print("=" * 60)

    ablation_config = [
        ('Proposed (Full)', ProposedModel),
        ('w/o Attention', AblationNoAttention),
        ('w/o Dual-Stream (Stream 1 only)', AblationStream1Only),
        ('w/o Dual-Stream (Stream 2 only)', AblationNoAttention)
    ]

    y_upper = y_all[:, 2]  # Upper FMA
    ablation_results = []

    for model_name, model_class in ablation_config:
        print(f"Evaluating variant: {model_name} ...")
        mean, std = train_and_evaluate(model_name, model_class, X, y_upper, FEAT_DIM, SEQ_LEN, epochs=EPOCHS)
        ablation_results.append({
            'Model': model_name,
            'Pearson r': f"{mean['r']:.2f}",
            'MAE': f"{mean['mae']:.1f}"
        })

    # ==========================================
    # 5. Generate LaTeX table (Output Latex)
    # ==========================================
    print("\n\n" + "=" * 60)
    print("📝  Latex  (Copy to paper)")
    print("=" * 60)

    # Table 1: Performance Comparison
    print("% Table 1: Performance comparison across different metrics")
    print("\\begin{table*}[ht]")
    print("\\centering")
    print("\\caption{Performance comparison of different models for Upper FMA, Total FMA, and ADL.}")
    print("\\label{tab:comparison}")
    print("\\renewcommand{\\arraystretch}{1.2}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print(
        " & \\multicolumn{2}{c}{\\textbf{Upper FMA}} & \\multicolumn{2}{c}{\\textbf{Total FMA}} & \\multicolumn{2}{c}{\\textbf{ADL}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    print(
        "\\textbf{Model} & \\textbf{$r$} & \\textbf{MAE} & \\textbf{$r$} & \\textbf{MAE} & \\textbf{$r$} & \\textbf{MAE} \\\\")
    print("\\midrule")

    for model_name, _ in models_config:
        row_str = f"{model_name}"
        for task in tasks:
            metrics = final_comparison_table[task][model_name]
            val_r = metrics['r']
            val_mae = metrics['mae']
            row_str += f" & {val_r:.4f} & {val_mae:.1f}"
        print(row_str + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

    print("\n")

    # Table 2: Ablation Study
    print("% Table 2: Ablation Study")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Ablation study of model components on Upper FMA.}")
    print("\\label{tab:ablation}")
    print("\\renewcommand{\\arraystretch}{1.2}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{Pearson $r$} & \\textbf{MAE} \\\\")
    print("\\midrule")

    for row in ablation_results:
        print(f"{row['Model']} & {row['Pearson r']} & {row['MAE']} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n✅ AllExperiment！LatexTables generated。")