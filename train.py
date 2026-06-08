import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. Data Loading and Preprocessing
def load_and_preprocess_data(data_dir, excel_path):
    """
    Load EEG data with preprocessing and feature selection
    """
    # Read Excel table
    df = pd.read_excel(excel_path)

    # Check and handle target columns
    target_columns = ['ADL', 'FMA', 'FMA-UE']

    # Create bins for stratified K-fold
    for col in target_columns:
        if col not in df.columns:
            print(f"Warning: column {col} not found in Excel")

    # Fill missing values with median
    for col in target_columns:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {col} 的 {df[col].isnull().sum()} missing values with median: {median_val:.2f}")

    # Prepare feature and target lists
    all_features = []
    all_targets = []
    valid_patient_ids = []
    valid_indices = []  # Track valid sample indices

    # Load H5 file for each patient
    current_idx = 0
    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        h5_file = os.path.join(data_dir, f"{patient_id}.h5")

        # Check if file exists
        if not os.path.exists(h5_file):
            print(f"File {h5_file} not found, skipping")
            continue

        try:
            # Load H5 file
            with h5py.File(h5_file, 'r') as f:
                # Get PSD features
                if 'psd_features' in f:
                    psd_features = f['psd_features'][:]
                else:
                    print(f"File {h5_file} does not contain 'psd_features' key, skipping")
                    continue

            # Validate feature shape (100, 29, 90)
            if psd_features.shape != (100, 29, 90):
                print(f"Warning: {patient_id} 的Feature shape为 {psd_features.shape}，not expected (100, 29, 90)")
                continue

            # Preserve temporal window structure - 转换为(100, 29*90)的特征序列
            time_features = psd_features.reshape(100, 29 * 90)

            # Normalize features per time window
            scaler = StandardScaler()
            time_features_scaled = scaler.fit_transform(time_features)

            # Collect features and targets
            all_features.append(time_features_scaled)
            targets = [row[col] if col in row else np.nan for col in target_columns]
            all_targets.append(targets)
            valid_patient_ids.append(patient_id)
            valid_indices.append(current_idx)

        except Exception as e:
            print(f"处理File {h5_file} 时出错: {e}")
            continue

        current_idx += 1

    # Convert to numpy arrays
    X = np.array(all_features)  # 形状: (samples数, 100, 2610)
    y = np.array(all_targets)

    print(f"Successfully loaded {len(X)} 个samples")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # 特征选择 - 降低维度
    print("\nPerforming feature selection and dimensionality reduction...")
    X_reshaped = X.reshape(X.shape[0], X.shape[1] * X.shape[2])  # (n_samples, 100*2610)

    # Combine targets for feature selection
    combined_targets = np.mean(y, axis=1)  # Mean of all targets as combined indicator

    # Select top1000 features
    selector = SelectKBest(f_regression, k=1000)
    X_selected_flat = selector.fit_transform(X_reshaped, combined_targets)

    # Reshape to temporal format (samples, 时间步, 特征)
    X_selected = X_selected_flat.reshape(X.shape[0], X.shape[1], -1)
    print(f"降维后Feature shape: {X_selected.shape}")

    # Create stratification labels
    y_stratify = np.digitize(combined_targets, np.percentile(combined_targets, [33, 66]))

    return X_selected, y, valid_patient_ids, df, target_columns, y_stratify, valid_indices


# 2. ESTAF-SMS: Spatiotemporal Attention Model
class ESTAF_SMS(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=3):
        super(ESTAF_SMS, self).__init__()
        """
        ESTAF-SMS Model Architecture:
        1. 1D CNN for local temporal features
        2. LSTM for long-range dependencies
        3. Attention mechanism for temporal importance
        4. Global pooling for robustness
        """
        self.input_size = input_size

        # 1D CNN用于提取局部特征
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

        # Feature fusion after global pooling
        self.global_pool_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # 拼接平均和最大池化
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights以确保稳定训练"""
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
        """
        x: (batch_size, seq_len=100, input_size)
        """
        batch_size = x.size(0)

        # 1. 转置以适应CNN: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # 2. CNN特征提取和降采样
        x = self.conv1(x)  # (batch, 16, 50)

        # 3. 转回LSTM格式: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, 50, 16)

        # 4. LSTM处理
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: (batch, 50, hidden_size)

        # 5. 注意力机制
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, 50, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)

        # 6. 全局池化 (平均 + 最大)
        global_avg_pool = torch.mean(lstm_out, dim=1)
        global_max_pool = torch.max(lstm_out, dim=1)[0]
        global_pooled = torch.cat([global_avg_pool, global_max_pool], dim=1)
        global_pooled = self.global_pool_layers(global_pooled)

        # 7. 融合注意力和全局池化特征
        fused_features = 0.7 * context + 0.3 * global_pooled

        # 8. Output layer
        output = self.output_layer(fused_features)

        return output


# 3. 数据增强
class EEGAugmentor:
    """EEG数据增强类"""

    @staticmethod
    def time_warp(x, sigma=0.2, knot=4):
        """时间扭曲增强 - 适用于时序数据"""
        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
        warp_steps = (np.linspace(0, x.shape[1] - 1., num=knot + 2)).astype(np.int64)

        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            random_warp = np.zeros((x.shape[1], x.shape[2]))
            for dim in range(x.shape[2]):
                random_warp[:, dim] = np.interp(orig_steps, warp_steps, random_warps[i, :, dim])

            # 应用扭曲
            for dim in range(x.shape[2]):
                result[i, :, dim] = np.interp(orig_steps, orig_steps * random_warp[:, dim], x[i, :, dim]).T

        return result

    @staticmethod
    def add_gaussian_noise(x, std=0.05):
        """添加高斯噪声"""
        noise = np.random.normal(0, std, x.shape)
        return x + noise

    @staticmethod
    def time_shift(x, max_shift=5):
        """时间平移"""
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                result[i, shift:, :] = x[i, :-shift, :]
                result[i, :shift, :] = x[i, 0, :]  # 用第一个时间点填充
            elif shift < 0:
                result[i, :shift, :] = x[i, -shift:, :]
                result[i, shift:, :] = x[i, -1, :]  # 用最后一个时间点填充
            else:
                result[i] = x[i]
        return result

    @staticmethod
    def augment(x, y, augmentation_factor=0.5):
        """应用多种增强方法"""
        if np.random.random() < augmentation_factor:
            # 随机选择增强方法
            aug_method = np.random.choice(['time_warp', 'noise', 'time_shift'])

            if aug_method == 'time_warp':
                return EEGAugmentor.time_warp(x, sigma=0.1)
            elif aug_method == 'noise':
                return EEGAugmentor.add_gaussian_noise(x, std=0.03)
            elif aug_method == 'time_shift':
                return EEGAugmentor.time_shift(x, max_shift=3)

        return x


class EEGDataset(Dataset):
    def __init__(self, features, targets, augment=False, augmentation_factor=0.5):
        # 确保特征和目标是numpy数组
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
        x = self.features[idx].copy()  # 避免修改原始数据
        y = self.targets[idx]

        if self.augment and np.random.random() < self.augmentation_factor:
            # 转换为numpy进行增强
            x_np = np.expand_dims(x, 0)  # 添加batch维度
            x_aug = EEGAugmentor.augment(x_np, y, augmentation_factor=1.0)[0]
            x = x_aug

        # 转换为张量
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


# 5. 改进的训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=25):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
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

        # 验证阶段
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

        # 学习率调度
        scheduler.step(val_loss)

        # 早停机制
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

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def evaluate_model(model, data_loader, target_columns):
    """全面Evaluate model性能"""
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

        # 计算多种评估指标
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
    # 加载和预处理数据
    X, y, patient_ids, df, target_columns, y_stratify, valid_indices = load_and_preprocess_data(data_dir, excel_path)

    # 5折分层交叉验证
    k_folds = 5
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = {col: [] for col in target_columns}
    fold_metrics = {col: {'mae': [], 'rmse': []} for col in target_columns}
    best_overall_model = None
    best_overall_score = -float('inf')
    best_fold = -1

    # 用于Save best model预测
    all_patient_predictions = np.zeros((X.shape[0], len(target_columns)))

    # 保存每折的验证数据和预测结果
    fold_val_data = {}
    fold_val_features = {}
    fold_val_indices = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_stratify)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{k_folds}")
        print(f"{'=' * 60}")

        print(f"训练集samples数: {len(train_idx)}, 验证集samples数: {len(val_idx)}")

        # 创建数据集和加载器
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 创建数据集
        train_dataset = EEGDataset(
            X_train, y_train,
            augment=True,
            augmentation_factor=0.7
        )

        val_dataset = EEGDataset(X_val, y_val, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        input_size = X.shape[2]
        model = ESTAF_SMS(
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

        # 保存验证特征和索引
        fold_val_features[fold] = X_val.copy()
        fold_val_indices[fold] = val_idx.copy()

        # Save best model
        if fold_score > best_overall_score:
            best_overall_score = fold_score
            best_overall_model = deepcopy(model)
            best_fold = fold + 1
            torch.save(model.state_dict(), f'best_fold_{fold + 1}_model.pth')
            print(f"New best model保存为 fold {fold + 1}, Average score: {fold_score:.4f}")

        # 保存当前fold的预测结果
        fold_indices = np.array(valid_indices)[val_idx]
        for i, idx in enumerate(fold_indices):
            all_patient_predictions[idx] = val_preds[i]

        # 保存fold结果
        for col in target_columns:
            r = fold_eval_results[col]['pearson_r']
            mae = fold_eval_results[col]['mae']
            rmse = fold_eval_results[col]['rmse']

            fold_results[col].append(r)
            fold_metrics[col]['mae'].append(mae)
            fold_metrics[col]['rmse'].append(rmse)

    # 打印5折交叉验证结果
    print("\n" + "=" * 50)
    print("5折交叉验证结果:")
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

    print(f"\n最佳模型来自 fold {best_fold}，Average score: {best_overall_score:.4f}")

    # 使用最佳fold的数据绘制散点图
    print(f"\n绘制最佳fold({best_fold})的散点图...")
    best_fold_data = fold_val_data[best_fold - 1]
    targets = best_fold_data['targets']
    predictions = best_fold_data['predictions']

    # 创建目录
    os.makedirs('scatter_plots', exist_ok=True)

    for i, col in enumerate(target_columns):
        true_vals = targets[:, i]
        pred_vals = predictions[:, i]

        # 计算回归线
        slope, intercept, r_value, p_value, std_err = stats.linregress(true_vals, pred_vals)
        line_x = np.array([true_vals.min(), true_vals.max()])
        line_y = slope * line_x + intercept

        # 绘制散点图和拟合直线
        plt.figure(figsize=(8, 8))
        plt.scatter(true_vals, pred_vals, alpha=0.7, s=50, edgecolors='k', label='Data Points')
        plt.plot(line_x, line_y, color='red', linewidth=2, label=f'Fitted Line (r={r_value:.4f})')

        plt.title(f'{col} - True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # 添加统计信息文本框
        textstr = f'Pearson r = {r_value:.4f}\np-value = {p_value:.4f}\nMAE = {np.mean(np.abs(true_vals - pred_vals)):.2f}\nRMSE = {np.sqrt(np.mean((true_vals - pred_vals) ** 2)):.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # 保存图片
        safe_col_name = col.replace('/', '_').replace('\\', '_')
        plt.savefig(f'scatter_plots/{safe_col_name}_best_fold_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("Scatter plots saved to scatter_plots/")

    # 预测缺失values（Using best model）
    print("\n" + "=" * 50)
    print("Predict missing values and fill original table...")
    print("=" * 50)

    best_overall_model.eval()

    # 为所有samples生成预测
    full_dataset = EEGDataset(X, y, augment=False)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=0)

    with torch.no_grad():
        all_predictions = []
        for features, _ in full_loader:
            features = features.to(device)
            outputs = best_overall_model(features)
            all_predictions.append(outputs.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    result_df = df.copy()

    # Fill prediction results
    for i, col in enumerate(target_columns):
        if col not in result_df.columns:
            print(f"Warning: column {col} 不存在于结果DataFrame中，跳过")
            continue

        # Create prediction column
        pred_col_name = f'pred_{col}'
        result_df[pred_col_name] = np.nan

        # 为所有有效samples设置预测values
        for j, idx in enumerate(valid_indices):
            if idx < len(result_df):
                result_df.at[idx, pred_col_name] = all_predictions[j, i]

        # Replace only missing values in original data
        missing_mask = result_df[col].isna()
        if missing_mask.any():
            print(f"Filled {missing_mask.sum()} missing {col} values")
            # 仅在缺失位置使用预测values
            result_df.loc[missing_mask, col] = result_df.loc[missing_mask, pred_col_name]

    # Save results
    output_file = 'predictions_with_missing_values_filled.xlsx'
    result_df.to_excel(output_file, index=False)
    print(f"Predictions saved to '{output_file}'")

    # Save best model
    model_file = 'final_best_model.pth'
    torch.save(best_overall_model.state_dict(), model_file)
    print(f"Best model saved as '{model_file}'")

    # Save predictions
    np.save('all_patient_predictions.npy', all_patient_predictions)
    print("All patient predictions saved as 'all_patient_predictions.npy'")

    return best_overall_model, result_df, all_patient_predictions


# Usage
if __name__ == "__main__":
    # Configure paths
    data_directory = "h5_files"  # H5 files directory
    excel_file_path = "总表.xlsx"  # Excel file path

    # Check if paths exist
    if not os.path.exists(data_directory):
        print(f"Warning: data directory '{data_directory}' not found, using relative path")

    if not os.path.exists(excel_file_path):
        print(f"Error: Excel file '{excel_file_path}' not found, check path")
        exit(1)

    # 运行Main function
    try:
        model, results_df, predictions = main(data_directory, excel_file_path)
        print("\nProgram completed successfully!")
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback

        traceback.print_exc()
