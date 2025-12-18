import os
import h5py
import numpy as np
import pandas as pd
import shap
import random
from torch.autograd import Variable
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_preprocess_data(data_dir, excel_path):
    df = pd.read_excel(excel_path)

    target_columns = ['ADL', 'FMA', 'FMA-UE']

    for col in target_columns:
        if col not in df.columns:
            print(f"Warning: Column {col} does not exist in Excel file")
            target_columns.remove(col)
    
    initial_count = len(df)
    df = df.dropna(subset=target_columns)
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} samples with missing target values, {len(df)} samples remaining")

    all_features = []
    all_targets = []
    valid_patient_ids = []
    valid_indices = []

    current_idx = 0
    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        h5_file = os.path.join(data_dir, f"{patient_id}.h5")

        if not os.path.exists(h5_file):
            print(f"File {h5_file} does not exist, skipping")
            continue

        try:
            with h5py.File(h5_file, 'r') as f:
                if 'psd_features' in f:
                    psd_features = f['psd_features'][:]
                else:
                    print(f"Key 'psd_features' does not exist in file {h5_file}, skipping")
                    continue

            if psd_features.shape != (100, 29, 90):
                print(f"Warning: Feature shape for {patient_id} is {psd_features.shape}, expected (100, 29, 90)")
                continue

            time_features = psd_features.reshape(100, 29 * 90)

            scaler = StandardScaler()
            time_features_scaled = scaler.fit_transform(time_features)

            all_features.append(time_features_scaled)
            targets = [row[col] for col in target_columns]
            all_targets.append(targets)
            valid_patient_ids.append(patient_id)
            valid_indices.append(current_idx)

        except Exception as e:
            print(f"Error processing file {h5_file}: {e}")
            continue

        current_idx += 1

    X = np.array(all_features)
    y = np.array(all_targets)

    print(f"Successfully loaded {len(X)} samples with complete data")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nPerforming feature selection for dimensionality reduction...")
    X_reshaped = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    combined_targets = np.mean(y, axis=1)

    selector = SelectKBest(f_regression, k=1000)
    X_selected_flat = selector.fit_transform(X_reshaped, combined_targets)

    X_selected = X_selected_flat.reshape(X.shape[0], X.shape[1], -1)
    print(f"Feature shape after dimensionality reduction: {X_selected.shape}")

    y_stratify = np.digitize(combined_targets, np.percentile(combined_targets, [33, 66]))

    return X_selected, y, valid_patient_ids, df, target_columns, y_stratify, valid_indices

class EnhancedTemporalModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=3):
        super(EnhancedTemporalModel, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

        self.global_pool_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = x.transpose(1, 2)

        lstm_out, (h_n, _) = self.lstm(x)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        global_avg_pool = torch.mean(lstm_out, dim=1)
        global_max_pool = torch.max(lstm_out, dim=1)[0]
        global_pooled = torch.cat([global_avg_pool, global_max_pool], dim=1)
        global_pooled = self.global_pool_layers(global_pooled)

        fused_features = 0.7 * context + 0.3 * global_pooled

        output = self.output_layer(fused_features)

        return output

class EEGDataAugmentor:
    @staticmethod
    def time_warp(x, sigma=0.2, knot=4):
        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
        warp_steps = (np.linspace(0, x.shape[1] - 1., num=knot + 2)).astype(np.int64)

        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            random_warp = np.zeros((x.shape[1], x.shape[2]))
            for dim in range(x.shape[2]):
                random_warp[:, dim] = np.interp(orig_steps, warp_steps, random_warps[i, :, dim])

            for dim in range(x.shape[2]):
                result[i, :, dim] = np.interp(orig_steps, orig_steps * random_warp[:, dim], x[i, :, dim]).T

        return result

    @staticmethod
    def add_gaussian_noise(x, std=0.05):
        noise = np.random.normal(0, std, x.shape)
        return x + noise

    @staticmethod
    def time_shift(x, max_shift=5):
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                result[i, shift:, :] = x[i, :-shift, :]
                result[i, :shift, :] = x[i, 0, :]
            elif shift < 0:
                result[i, :shift, :] = x[i, -shift:, :]
                result[i, shift:, :] = x[i, -1, :]
            else:
                result[i] = x[i]
        return result

    @staticmethod
    def augment(x, y, augmentation_factor=0.5):
        if np.random.random() < augmentation_factor:
            aug_method = np.random.choice(['time_warp', 'noise', 'time_shift'])

            if aug_method == 'time_warp':
                return EEGDataAugmentor.time_warp(x, sigma=0.1)
            elif aug_method == 'noise':
                return EEGDataAugmentor.add_gaussian_noise(x, std=0.03)
            elif aug_method == 'time_shift':
                return EEGDataAugmentor.time_shift(x, max_shift=3)

        return x

class AugmentedEEGDataset(Dataset):
    def __init__(self, features, targets, augment=False, augmentation_factor=0.5):
        if isinstance(features, torch.Tensor):
            self.features = features.numpy()
        else:
            self.features = np.array(features)

        if isinstance(targets, torch.Tensor):
            self.targets = targets.numpy()
        else:
            self.targets = np.array(targets)

        self.augment = augment
        self.augmentation_factor = augmentation_factor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].copy()
        y = self.targets[idx]

        if self.augment and np.random.random() < self.augmentation_factor:
            x_np = np.expand_dims(x, 0)
            x_aug = EEGDataAugmentor.augment(x_np, y, augmentation_factor=1.0)[0]
            x = x_aug

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=25):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

def evaluate_model(model, data_loader, target_columns):
    model.eval()
    predictions = []
    targets_list = []

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets_list)

    results = {}
    for i, col in enumerate(target_columns):
        pred = predictions[:, i]
        true = targets[:, i]

        r, p_val = stats.pearsonr(true, pred)
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred) ** 2))

        results[col] = {
            'pearson_r': r,
            'p_value': p_val,
            'mae': mae,
            'rmse': rmse,
            'predictions': pred,
            'targets': true
        }

        print(f"{col} - Pearson r: {r:.4f} (p={p_val:.4f}), MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return results, predictions, targets

def main(data_dir, excel_path):
    X, y, patient_ids, df, target_columns, y_stratify, valid_indices = load_and_preprocess_data(data_dir, excel_path)

    k_folds = 5
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = {col: [] for col in target_columns}
    fold_metrics = {col: {'mae': [], 'rmse': []} for col in target_columns}
    best_overall_model = None
    best_overall_score = -float('inf')
    best_fold = -1

    all_patient_predictions = np.zeros((X.shape[0], len(target_columns)))

    fold_val_data = {}
    fold_val_features = {}
    fold_val_indices = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_stratify)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{k_folds}")
        print(f"{'=' * 60}")

        print(f"Training set size: {len(train_idx)}, Validation set size: {len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = AugmentedEEGDataset(
            X_train, y_train,
            augment=True,
            augmentation_factor=0.7
        )

        val_dataset = AugmentedEEGDataset(X_val, y_val, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        input_size = X.shape[2]
        model = EnhancedTemporalModel(
            input_size=input_size,
            hidden_size=32,
            output_size=len(target_columns)
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.005,
            weight_decay=1e-3,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            epochs=150, patience=20
        )

        fold_eval_results, val_preds, val_targets = evaluate_model(model, val_loader, target_columns)

        fold_score = sum([fold_eval_results[col]['pearson_r'] for col in target_columns]) / len(target_columns)

        fold_val_data[fold] = {
            'targets': val_targets,
            'predictions': val_preds
        }

        fold_val_features[fold] = X_val.copy()
        fold_val_indices[fold] = val_idx.copy()

        if fold_score > best_overall_score:
            best_overall_score = fold_score
            best_overall_model = deepcopy(model)
            best_fold = fold + 1
            torch.save(model.state_dict(), f'best_fold_{fold + 1}_model.pth')
            print(f"New best model saved from fold {fold + 1}, overall score: {fold_score:.4f}")

        fold_indices = np.array(valid_indices)[val_idx]
        for i, idx in enumerate(fold_indices):
            all_patient_predictions[idx] = val_preds[i]

        for col in target_columns:
            r = fold_eval_results[col]['pearson_r']
            mae = fold_eval_results[col]['mae']
            rmse = fold_eval_results[col]['rmse']

            fold_results[col].append(r)
            fold_metrics[col]['mae'].append(mae)
            fold_metrics[col]['rmse'].append(rmse)

    print("\n" + "=" * 50)
    print("5-Fold Cross-Validation Results (Complete Data Only):")
    print("=" * 50)
    for col in target_columns:
        mean_r = np.mean(fold_results[col])
        std_r = np.std(fold_results[col])
        mean_mae = np.mean(fold_metrics[col]['mae'])
        mean_rmse = np.mean(fold_metrics[col]['rmse'])

        print(f"{col}:")
        print(f"  Pearson r = {mean_r:.4f} ± {std_r:.4f}")
        print(f"  MAE = {mean_mae:.2f} ± {np.std(fold_metrics[col]['mae']):.2f}")
        print(f"  RMSE = {mean_rmse:.2f} ± {np.std(fold_metrics[col]['rmse']):.2f}")

    print(f"\nBest model from fold {best_fold}, overall score: {best_overall_score:.4f}")

    print("\n" + "=" * 50)
    print("Generating predictions for all complete samples...")
    print("=" * 50)

    best_overall_model.eval()

    full_dataset = AugmentedEEGDataset(X, y, augment=False)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=0)

    with torch.no_grad():
        all_predictions = []
        for features, _ in full_loader:
            features = features.to(device)
            outputs = best_overall_model(features)
            all_predictions.append(outputs.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    
    result_df = df.iloc[:len(X)].copy()
    for i, col in enumerate(target_columns):
        pred_col_name = f'pred_{col}'
        result_df[pred_col_name] = all_predictions[:, i]

    output_file = 'predictions_complete_data.xlsx'
    result_df.to_excel(output_file, index=False)
    print(f"Prediction results (complete data only) saved to '{output_file}'")

    model_file = 'final_best_model_complete_data.pth'
    torch.save(best_overall_model.state_dict(), model_file)
    print(f"Best model (trained on complete data) saved as '{model_file}'")

    np.save('all_patient_predictions_complete_data.npy', all_predictions)
    print("All patient predictions (complete data) saved as 'all_patient_predictions_complete_data.npy'")

    return best_overall_model, result_df, all_predictions

if __name__ == "__main__":
    data_directory = "path/to/h5_files"
    excel_file_path = "path/to/your_data.xlsx"

    if not os.path.exists(data_directory):
        print(f"Warning: Data directory '{data_directory}' does not exist, using relative path")

    if not os.path.exists(excel_file_path):
        print(f"Error: Excel file '{excel_file_path}' does not exist, please check the path")
        exit(1)

    try:
        model, results_df, predictions = main(data_directory, excel_file_path)
        print("\nProgram completed successfully! (Only complete data used)")
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()
