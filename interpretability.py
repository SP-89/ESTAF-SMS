import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mne  # 必须安装: pip install mne
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

# ==========================================
# 设置随机种子
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==========================================
# 全局绘图风格 (字体大幅调大，适配 Word/PPT)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 14,  # 基础字号
    'axes.labelsize': 18,  # 轴标签
    'axes.titlesize': 20,  # 标题
    'axes.titleweight': 'bold',  # 标题加粗
    'xtick.labelsize': 15,  # X轴刻度
    'ytick.labelsize': 15,  # Y轴刻度
    'legend.fontsize': 14,  # 图例
    'figure.titlesize': 22,  # 画布标题
    'figure.dpi': 300,  # 高DPI
    'axes.facecolor': 'white',
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,  # 线条加粗
    'axes.linewidth': 1.5  # 坐标轴线加粗
})


# ==========================================
# 1. 数据加载
# ==========================================
def load_eeg_data(data_dir, excel_path):
    df = pd.read_excel(excel_path)
    target_columns = ['ADL', 'FMA', 'FMA-UE']

    for col in target_columns:
        if col not in df.columns: print(f"警告: 列 {col} 不存在")

    for col in target_columns:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"已填充 {col} 的缺失值")

    all_features = []
    all_targets = []
    valid_patient_ids = []
    valid_indices = []

    current_idx = 0
    n_channels = 29
    n_freqs = 90

    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        h5_file = os.path.join(data_dir, f"{patient_id}.h5")
        if not os.path.exists(h5_file): continue

        try:
            with h5py.File(h5_file, 'r') as f:
                if 'psd_features' in f:
                    psd_features = f['psd_features'][:]
                else:
                    continue

            if psd_features.shape != (100, 29, 90): continue

            time_features = psd_features.reshape(100, 29 * 90)
            scaler = StandardScaler()
            time_features_scaled = scaler.fit_transform(time_features)

            all_features.append(time_features_scaled)
            targets = [row[col] if col in row else np.nan for col in target_columns]
            all_targets.append(targets)
            valid_patient_ids.append(patient_id)
            valid_indices.append(current_idx)
        except Exception:
            continue
        current_idx += 1

    X = np.array(all_features)
    y = np.array(all_targets)

    print(f"成功加载 {len(X)} 个样本")

    # --- 特征选择 ---
    print("\n正在进行特征选择降维...")
    X_reshaped = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    combined_targets = np.mean(y, axis=1)

    selector = SelectKBest(f_regression, k=1000)
    X_selected_flat = selector.fit_transform(X_reshaped, combined_targets)
    feature_mask = selector.get_support()
    X_selected = X_selected_flat.reshape(X.shape[0], X.shape[1], -1)

    y_stratify = np.digitize(combined_targets, np.percentile(combined_targets, [33, 66]))

    map_info = {
        'mask': feature_mask,
        'n_channels': n_channels,
        'n_freqs': n_freqs
    }

    return X_selected, y, valid_patient_ids, df, target_columns, y_stratify, valid_indices, map_info


# ==========================================
# 2. 模型与工具类
# ==========================================
class ESTAF_SMS(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=3):
        super(ESTAF_SMS, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(16, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.attention = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.global_pool_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(0.3)
        )
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        global_avg = torch.mean(lstm_out, dim=1)
        global_max = torch.max(lstm_out, dim=1)[0]
        fused = 0.7 * context + 0.3 * self.global_pool_layers(torch.cat([global_avg, global_max], dim=1))
        return self.output_layer(fused)


class EEGAugmentor:
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
        return x + np.random.normal(0, std, x.shape)

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
            aug = np.random.choice(['time_warp', 'noise', 'time_shift'])
            if aug == 'time_warp':
                return EEGAugmentor.time_warp(x, sigma=0.1)
            elif aug == 'noise':
                return EEGAugmentor.add_gaussian_noise(x, std=0.03)
            elif aug == 'time_shift':
                return EEGAugmentor.time_shift(x, max_shift=3)
        return x


class EEGDataset(Dataset):
    def __init__(self, features, targets, augment=False, augmentation_factor=0.5):
        self.features = features if isinstance(features, np.ndarray) else features.numpy()
        self.targets = targets if isinstance(targets, np.ndarray) else targets.numpy()
        self.augment = augment
        self.augmentation_factor = augmentation_factor

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        x, y = self.features[idx].copy(), self.targets[idx]
        if self.augment and np.random.random() < self.augmentation_factor:
            x = EEGAugmentor.augment(np.expand_dims(x, 0), y, 1.0)[0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=25):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for fx, fy in train_loader:
            fx, fy = fx.to(device), fy.to(device)
            optimizer.zero_grad()
            loss = criterion(model(fx), fy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fx, fy in val_loader:
                val_loss += criterion(model(fx.to(device)), fy.to(device)).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state: model.load_state_dict(best_state)
    return model, train_losses, val_losses


def evaluate_model(model, data_loader, target_columns):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for fx, fy in data_loader:
            preds.append(model(fx.to(device)).cpu().numpy())
            trues.append(fy.cpu().numpy())
    preds, trues = np.vstack(preds), np.vstack(trues)

    results = {}
    for i, col in enumerate(target_columns):
        r, p = stats.pearsonr(trues[:, i], preds[:, i])
        mae = np.mean(np.abs(trues[:, i] - preds[:, i]))
        rmse = np.sqrt(np.mean((trues[:, i] - preds[:, i]) ** 2))
        results[col] = {'pearson_r': r, 'p_value': p, 'mae': mae, 'rmse': rmse}
        print(f"{col} - Pearson r: {r:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return results, preds, trues


# ==========================================================
# [修改] 绘制带 P 值标注的散点图函数 (去除理想对角线)
# ==========================================================
def plot_scatter(y_true, y_pred, target_name, save_dir='scatter_plots'):
    """
    绘制真实值 vs 预测值散点图，并叠加拟合回归线（反映相关系数趋势）。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 计算统计量
    r, p = stats.pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))

    # 2. 准备绘图
    plt.figure(figsize=(8, 7))

    # [修改核心] 使用 regplot 替代 scatterplot
    sns.regplot(
        x=y_true,
        y=y_pred,
        color='#2878B5',  # 主色调
        # --- 修正点在此处 ---
        # 注意：这里必须用 linewidths (复数)，否则会报错 TypeError
        scatter_kws={'s': 100, 'alpha': 0.7, 'edgecolor': 'k', 'linewidths': 0.8},
        # ------------------
        line_kws={'color': '#D62728', 'linewidth': 3},  # 回归线样式 (红色粗线)
        ci=None
    )

    # 3. 构造标注文本逻辑
    p_text = "$p < 0.05$" if p < 0.05 else f"$p = {p:.3f}$"
    stats_text = (f"Pearson $R = {r:.3f}$\n"
                  f"{p_text}\n"
                  f"MAE $= {mae:.2f}$")

    # 4. 添加文本框 (字体设为18)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1)
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=18,
             verticalalignment='top', bbox=props)

    plt.title(f"{target_name}: True vs Predicted", pad=15)
    plt.xlabel(f"True {target_name}")
    plt.ylabel(f"Predicted {target_name}")

    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(save_dir, f'scatter_best_fold_{target_name}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"已保存最佳Fold散点图 (含回归线): {save_path}")


# ==========================================================
# SHAP 可解释性分析类
# ==========================================================
class SHAPAnalyzer:
    def __init__(self, model, train_data_sample, map_info, channel_names=None):
        self.map_info = map_info
        self.n_channels = map_info['n_channels']
        self.n_freqs = map_info['n_freqs']
        self.mask = map_info['mask']

        if channel_names is None or len(channel_names) != self.n_channels:
            print(f"[SHAP] 通道名数量不符，使用默认名称。")
            self.channel_names = [f'Ch{i + 1}' for i in range(self.n_channels)]
        else:
            self.channel_names = channel_names

        # 解码特征名
        self.feature_names_decoded = []
        original_indices = np.where(self.mask)[0]
        features_per_step = self.n_channels * self.n_freqs
        for idx in original_indices:
            within_step_idx = idx % features_per_step
            ch_idx = int(within_step_idx // self.n_freqs)
            freq_idx = int(within_step_idx % self.n_freqs)
            actual_freq = freq_idx * 0.5  # 0-89 index -> 0-45 Hz

            if ch_idx < len(self.channel_names):
                feat_name = f"{self.channel_names[ch_idx]} ({actual_freq:.1f}Hz)"
            else:
                feat_name = f"Unknown_Ch{ch_idx}"
            self.feature_names_decoded.append(feat_name)

        print("\n[SHAP] 初始化解释器 (CPU模式)...")
        self.device = torch.device('cpu')
        self.model = deepcopy(model).to(self.device)
        self.train_data = train_data_sample.to(self.device)

        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.BatchNorm1d):
                module.eval()

        if not self.train_data.requires_grad: self.train_data.requires_grad = True
        self.explainer = shap.GradientExplainer(self.model, self.train_data)

    def analyze(self, test_data_sample, target_columns, save_dir='shap_results'):
        os.makedirs(save_dir, exist_ok=True)
        test_data_sample = test_data_sample.to(self.device)
        if not test_data_sample.requires_grad: test_data_sample.requires_grad = True

        print("[SHAP] 正在计算 SHAP 值 (可能较慢)...")
        shap_values_list = self.explainer.shap_values(test_data_sample)

        if not isinstance(shap_values_list, list):
            if len(shap_values_list.shape) == 4:
                temp = []
                for i in range(shap_values_list.shape[-1]): temp.append(shap_values_list[..., i])
                shap_values_list = temp
            else:
                shap_values_list = [shap_values_list]

        shap_abs_combined = np.sum([np.abs(x) for x in shap_values_list], axis=0)
        shap_flat = shap_abs_combined.reshape(shap_abs_combined.shape[0], -1)
        total_imp = np.sum(shap_flat, axis=0)

        full_flat_imp = np.zeros(100 * self.n_channels * self.n_freqs)
        full_flat_imp[self.mask] = total_imp
        full_volume = full_flat_imp.reshape(100, self.n_channels, self.n_freqs)
        psd_imp_matrix = np.sum(full_volume, axis=0)

        # --- 绘图 (注重清晰度) ---
        print("[SHAP] 正在生成高清图表...")

        # A. 地形图
        channel_scores = np.sum(psd_imp_matrix, axis=1)
        channel_scores_norm = channel_scores / (np.max(channel_scores) + 1e-9)
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(ch_names=self.channel_names, sfreq=100, ch_types='eeg')
            info.set_montage(montage, match_case=False, on_missing='raise')
            fake_evoked = mne.EvokedArray(channel_scores_norm[:, np.newaxis], info)

            fig, ax = plt.subplots(figsize=(8, 7))  # 加大画布
            mne.viz.plot_topomap(
                fake_evoked.data[:, 0], fake_evoked.info, axes=ax, cmap='Spectral_r',
                names=self.channel_names, sensors=True, outlines='head',
                contours=6, image_interp='cubic', show=False
            )
            sm = plt.cm.ScalarMappable(cmap='Spectral_r', norm=plt.Normalize(0, 1))
            cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
            cbar.set_label('Normalized Importance', size=16, weight='bold')
            cbar.ax.tick_params(labelsize=14)

            ax.set_title('Spatial Importance Topomap', pad=25, fontweight='bold', fontsize=20)
            plt.savefig(os.path.join(save_dir, 'spatial_topomap.png'), bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            print(f"地形图生成失败: {e}")

        # B. 频段重要性
        bands = {
            'Delta\n(0-4Hz)': slice(0, 8),
            'Theta\n(4-8Hz)': slice(8, 16),
            'Alpha\n(8-13Hz)': slice(16, 26),
            'Beta\n(13-30Hz)': slice(26, 60),
            'Low Gamma\n(30-45Hz)': slice(60, 90)
        }
        band_vals = [np.sum(psd_imp_matrix[:, v]) for v in bands.values()]
        band_vals = np.array(band_vals) / (np.max(band_vals) + 1e-9)

        fig, ax = plt.subplots(figsize=(10, 6))  # 加大画布
        sns.barplot(x=list(bands.keys()), y=band_vals, palette="rocket", ax=ax, edgecolor='k', linewidth=1.5)
        ax.set_ylabel('Normalized Importance', fontsize=18)
        ax.set_xlabel('Frequency Bands', fontsize=18)
        plt.xticks(fontsize=15, rotation=0)
        plt.yticks(fontsize=15)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'spectral_importance.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # C. 频谱热力图
        top_idx = np.argsort(channel_scores)[::-1][:10]
        fig, ax = plt.subplots(figsize=(12, 6))  # 加宽画布

        sns.heatmap(psd_imp_matrix[top_idx, :], cmap='viridis',
                    yticklabels=[self.channel_names[i] for i in top_idx], rasterized=True,
                    cbar_kws={'label': 'Importance Score'})

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Importance', size=16)

        xticks = np.arange(0, 91, 10)
        xticklabels = [int(x * 0.5) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, fontsize=14)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15, rotation=0)

        ax.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
        ax.set_title('Top 10 Channels Spectral Signature', fontsize=20, pad=15)
        plt.savefig(os.path.join(save_dir, 'channel_freq_heatmap.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # D. Summary Plot
        X_flat = test_data_sample.detach().cpu().numpy().reshape(test_data_sample.shape[0], -1)
        for i, target in enumerate(target_columns):
            if i >= len(shap_values_list): break
            shap_t = shap_values_list[i].reshape(shap_values_list[i].shape[0], -1)

            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_t, X_flat, feature_names=self.feature_names_decoded,
                              show=False, max_display=10, plot_size=(10, 8))

            ax = plt.gca()
            ax.set_xlabel('SHAP value (impact on model output)', fontsize=16)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=14)
            plt.title(f'Top Predictors for {target}', fontsize=20, pad=20, fontweight='bold')

            if len(fig.axes) > 1:
                cbar_ax = fig.axes[1]
                cbar_ax.tick_params(labelsize=14)
                cbar_ax.set_ylabel('Feature Value', fontsize=16)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'summary_{target}.png'), dpi=300)
            plt.close()

        return [self.channel_names[i] for i in top_idx[:3]]


def main(data_dir, excel_path):
    X, y, patient_ids, df, target_columns, y_stratify, valid_indices, map_info = load_eeg_data(data_dir,
                                                                                                          excel_path)

    k_folds = 5
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = {col: [] for col in target_columns}
    best_overall_score = -float('inf')
    best_overall_model = None
    best_fold = -1
    all_patient_predictions = np.zeros((X.shape[0], len(target_columns)))

    # 存储用于解释和绘图的数据
    best_train_data = None
    fold_val_data_best = None

    # [新增] 专门存储最佳fold的预测结果和真实值
    best_fold_preds = None
    best_fold_trues = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_stratify)):
        print(f"\nFold {fold + 1}/{k_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_loader = DataLoader(EEGDataset(X_train, y_train, True, 0.7), batch_size=16, shuffle=True)
        val_loader = DataLoader(EEGDataset(X_val, y_val, False), batch_size=32, shuffle=False)

        model = ESTAF_SMS(input_size=X.shape[2], output_size=len(target_columns)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        model, _, _ = train_model(model, train_loader, val_loader, nn.MSELoss(), optimizer, scheduler, epochs=150,
                                  patience=20)

        eval_res, preds, trues = evaluate_model(model, val_loader, target_columns)

        fold_score = sum([eval_res[col]['pearson_r'] for col in target_columns]) / len(target_columns)

        if fold_score > best_overall_score:
            best_overall_score = fold_score
            best_overall_model = deepcopy(model)
            best_fold = fold + 1
            best_train_data = X_train
            fold_val_data_best = X_val

            # [新增] 保存最佳Fold的预测结果用于绘图
            best_fold_preds = preds
            best_fold_trues = trues

            torch.save(model.state_dict(), f'best_fold_{fold + 1}_model.pth')
            print(f"新最佳模型: Fold {fold + 1}, Score: {fold_score:.4f}")

        fold_idx_global = np.array(valid_indices)[val_idx]
        for i, idx in enumerate(fold_idx_global): all_patient_predictions[idx] = preds[i]
        for col in target_columns: fold_results[col].append(eval_res[col]['pearson_r'])

    print(f"\n最佳Fold: {best_fold}, Score: {best_overall_score:.4f}")

    # ==========================================================
    # [修改] 绘制最佳 Fold 的散点图
    # ==========================================================
    print("\n" + "=" * 50 + "\n绘制最佳Fold预测散点图\n" + "=" * 50)
    if best_fold_preds is not None and best_fold_trues is not None:
        for i, col in enumerate(target_columns):
            # 只使用 best_fold 的数据
            plot_scatter(best_fold_trues[:, i], best_fold_preds[:, i], col, save_dir='scatter_plots')

    # ==========================================================
    # SHAP 分析
    # ==========================================================
    print("\n" + "=" * 50 + "\n开始 SHAP 分析\n" + "=" * 50)
    bg_idx = np.random.choice(len(best_train_data), 50, replace=False)
    bg_tensor = torch.tensor(best_train_data[bg_idx], dtype=torch.float32)

    test_idx = np.random.choice(len(fold_val_data_best), min(20, len(fold_val_data_best)), replace=False)
    test_tensor = torch.tensor(fold_val_data_best[test_idx], dtype=torch.float32)

    real_channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'Oz', 'O2', 'Iz'
    ]

    analyzer = SHAPAnalyzer(best_overall_model, bg_tensor, map_info, channel_names=real_channel_names)
    top_channels = analyzer.analyze(test_tensor, target_columns)
    print(f"最重要通道: {top_channels}")

    # 保存全量预测表
    res_df = df.copy()
    for i, col in enumerate(target_columns):
        if col not in res_df: continue
        res_df[f'pred_{col}'] = np.nan
        for j, idx in enumerate(valid_indices): res_df.at[idx, f'pred_{col}'] = all_patient_predictions[j, i]
        mask = res_df[col].isna()
        if mask.any(): res_df.loc[mask, col] = res_df.loc[mask, f'pred_{col}']

    res_df.to_excel('predictions_filled.xlsx', index=False)
    return best_overall_model, res_df, all_patient_predictions


if __name__ == "__main__":
    data_dir = "h5_files"
    excel_path = "总表.xlsx"
    if os.path.exists(excel_path):
        main(data_dir, excel_path)
    else:
        print("文件不存在")