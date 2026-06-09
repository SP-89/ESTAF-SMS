"""
Published EEG baselines adapted for PSD regression.
EEGNet (Lawhern 2018) + DeepConvNet (Schirrmeister 2017)
Minimal but faithful implementations.
"""
import os, h5py, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
TARGET_COLS = ['ADL','FMA','FMA-UE']

# ====== DATA with band-power ======
BAND_SLICES = [slice(0,8),slice(8,16),slice(16,26),slice(26,60),slice(60,90)]
N_BANDS = len(BAND_SLICES)
def load_data(data_dir, excel_path):
    df = pd.read_excel(excel_path)
    for c in TARGET_COLS:
        if c in df and df[c].isnull().any(): df[c] = df[c].fillna(df[c].median())
    X_list, y_list = [], []
    for _, row in df.iterrows():
        pid = row['patient_id']
        h5f = os.path.join(data_dir, f'{pid}.h5')
        if not os.path.exists(h5f): continue
        try:
            with h5py.File(h5f,'r') as f:
                if 'psd_features' not in f: continue
                psd = f['psd_features'][:]
            if psd.shape != (100,29,90): continue
            bp = np.zeros((100,29,N_BANDS), dtype=np.float32)
            for b,sl in enumerate(BAND_SLICES): bp[:,:,b] = np.mean(psd[:,:,sl], axis=2)
            feats = bp.reshape(100, 29*N_BANDS).astype(np.float32)
            feats = StandardScaler().fit_transform(feats)
            X_list.append(feats)
            y_list.append([row.get(c,np.nan) for c in TARGET_COLS])
        except: continue
    X = np.array(X_list); y = np.array(y_list, dtype=np.float32)
    comb = np.mean(y, axis=1)
    ys = np.digitize(comb, np.percentile(comb, [33,66]))
    print(f'Loaded {len(X)} samples, X:{X.shape}')
    return X, y, ys

# ====== EEGNet adapted for PSD ======

class EEGNet_PSD(nn.Module):
    """EEGNet adapted for band-power PSD: input (B,100,145) reshaped to (B,5,29,100)"""
    def __init__(self, output_size=3, F1=8, D=2, F2=16):
        super().__init__()
        self.conv1 = nn.Conv2d(5, F1, (1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(F1)
        self.dw_conv = nn.Conv2d(F1, D*F1, (29, 1), groups=F1)
        self.bn_dw = nn.BatchNorm2d(D*F1)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.sep_conv = nn.Conv2d(D*F1, D*F1, (1, 16), padding=(0, 8), groups=D*F1)
        self.bn_sep = nn.BatchNorm2d(D*F1)
        self.pw_conv = nn.Conv2d(D*F1, F2, (1, 1))
        self.bn_pw = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)
        self.fc = None

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B, T, 29, 5).permute(0, 3, 2, 1)
        x = self.drop1(self.pool1(torch.relu(self.bn_dw(self.dw_conv(torch.relu(self.bn1(self.conv1(x))))))))
        x = self.drop2(self.pool2(torch.relu(self.bn_pw(self.pw_conv(torch.relu(self.bn_sep(self.sep_conv(x))))))))
        x = x.view(B, -1)
        if self.fc is None: self.fc = nn.Linear(x.size(1), 3).to(x.device)
        return self.fc(x)

class DeepConvNet_PSD(nn.Module):
    """DeepConvNet adapted for band-power PSD: input (B,100,145) reshaped to (B,5,29,100)"""
    def __init__(self, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 25, (1, 10))
        self.bn1 = nn.BatchNorm2d(25)
        self.conv2 = nn.Conv2d(25, 50, (29, 1))
        self.bn2 = nn.BatchNorm2d(50)
        self.pool1 = nn.MaxPool2d((1, 3))
        self.drop1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(50, 100, (1, 10))
        self.bn3 = nn.BatchNorm2d(100)
        self.conv4 = nn.Conv2d(100, 200, (1, 10))
        self.bn4 = nn.BatchNorm2d(200)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.drop2 = nn.Dropout(0.5)
        self.fc = None

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B, T, 29, 5).permute(0, 3, 2, 1)
        x = self.drop1(self.pool1(torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))))))
        x = self.drop2(self.pool2(torch.relu(self.bn4(self.conv4(torch.relu(self.bn3(self.conv3(x))))))))
        x = x.view(B, -1)
        if self.fc is None: self.fc = nn.Linear(x.size(1), 3).to(x.device)
        return self.fc(x)


# ====== TRAINING ======
class EEGDataset(Dataset):
    def __init__(self, X, y): self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_model(model, tr_ld, vl_ld, opt, sch, epochs=150, patience=20):
    criterion = nn.MSELoss()
    best_loss = float('inf'); best_state = None; pat = 0
    for ep in range(epochs):
        model.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(Xb), yb)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); vl = 0; n = 0
        with torch.no_grad():
            for Xb, yb in vl_ld:
                Xb, yb = Xb.to(device), yb.to(device)
                vl += criterion(model(Xb), yb).item() * len(Xb)
                n += len(Xb)
        vl /= n; sch.step(vl)
        if vl < best_loss: best_loss = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat = 0
        else: pat += 1
        if pat >= patience: break
    model.load_state_dict(best_state)
    return model

@torch.no_grad()
def evaluate(model, loader, cols):
    model.eval(); preds, trues = [], []
    for Xb, yb in loader:
        preds.append(model(Xb.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    preds = np.vstack(preds); trues = np.vstack(trues)
    res = {}
    for i, c in enumerate(cols):
        r, _ = stats.pearsonr(preds[:, i], trues[:, i])
        m = np.mean(np.abs(preds[:, i] - trues[:, i]))
        res[c] = {'r': r, 'mae': m}
    return res

def run_cv(model_cls, X, y, ys, cols, epochs=150, patience=20):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f_r = {c: [] for c in cols}
    for fold, (tr, vl) in enumerate(kf.split(X, ys)):
        Xt, Xv = X[tr], X[vl]; yt, yv = y[tr], y[vl]
        tr_ds = EEGDataset(Xt, yt); vl_ds = EEGDataset(Xv, yv)
        tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True)
        vl_ld = DataLoader(vl_ds, batch_size=32, shuffle=False)
        m = model_cls(output_size=len(cols)).to(device)
        opt = optim.AdamW(m.parameters(), lr=0.005, weight_decay=1e-3)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        m = train_model(m, tr_ld, vl_ld, opt, sch, epochs, patience)
        res = evaluate(m, vl_ld, cols)
        for c in cols: f_r[c].append(res[c]['r'])
    summ = {}
    for c in cols:
        summ[c] = {'mean_r': np.mean(f_r[c]), 'std_r': np.std(f_r[c])}
        print(f'  {c}: R={np.mean(f_r[c]):.4f}+/-{np.std(f_r[c]):.4f}')
    return summ

# ====== MAIN ======
if __name__ == '__main__':
    import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('Loading data...')
    X, y, ys = load_data('h5_files', '总表.xlsx')  # TODO: update path to your metadata file
    print(f'X: {X.shape}, y: {y.shape}')
    
    print('\n=== EEGNet (Lawhern et al., 2018) ===')
    eegnet_sum = run_cv(EEGNet_PSD, X, y, ys, TARGET_COLS)
    
    print('\n=== DeepConvNet (Schirrmeister et al., 2017) ===')
    dcn_sum = run_cv(DeepConvNet_PSD, X, y, ys, TARGET_COLS)
    
    print('\n=== RESULTS ===')
    for name, s in [('EEGNet', eegnet_sum), ('DeepConvNet', dcn_sum)]:
        print(f'{name}:')
        for c in TARGET_COLS:
            print(f'  {c}: R={s[c]["mean_r"]:.4f}+/-{s[c]["std_r"]:.4f}')
    print('\nDone!')
# ====== MODELS FOR PSD/SPECTRAL FEATURES ======

class FBCNet(nn.Module):
    """FBCNet-style (Mane et al., 2021) - spectral-spatial filtering for band-power"""
    def __init__(self, output_size=3, n_bands=5, n_ch=29):
        super().__init__()
        self.spatial_conv = nn.Conv2d(n_bands, 4*n_bands, (n_ch, 1), groups=n_bands)
        self.bn1 = nn.BatchNorm2d(4*n_bands)
        self.temp_conv = nn.Conv2d(4*n_bands, 8*n_bands, (1, 10), padding=(0,5))
        self.bn2 = nn.BatchNorm2d(8*n_bands)
        self.pool = nn.AvgPool2d((1, 5))
        self.drop = nn.Dropout(0.5)
        self.fc = None

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B, T, 29, 5).permute(0,3,2,1).contiguous()
        x = self.pool(torch.relu(self.bn2(self.temp_conv(torch.relu(self.bn1(self.spatial_conv(x)))))))
        x = self.drop(x).view(B, -1)
        if self.fc is None: self.fc = nn.Linear(x.size(1), 3).to(x.device)
        return self.fc(x)


class TSCeption(nn.Module):
    """TSception-style (Ding et al., 2022) - multi-scale temporal + spatial attention"""
    def __init__(self, output_size=3, n_ch=29):
        super().__init__()
        self.conv_s = nn.Conv2d(5, 8, (1, 15), padding=(0,7))
        self.conv_m = nn.Conv2d(5, 8, (1, 31), padding=(0,15))
        self.conv_l = nn.Conv2d(5, 8, (1, 63), padding=(0,31))
        self.bn = nn.BatchNorm2d(24)
        self.spatial_attn = nn.Sequential(nn.Conv2d(24, 1, (n_ch, 1)), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d((1, 16))
        self.drop = nn.Dropout(0.5)
        self.fc = None

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B, T, 29, 5).permute(0,3,2,1).contiguous()
        xs = self.conv_s(x); xm = self.conv_m(x); xl = self.conv_l(x)
        x = torch.cat([xs, xm, xl], dim=1)
        x = self.bn(x)
        attn = self.spatial_attn(x)
        x = (x * attn)
        x = self.pool(x).view(B, -1)
        x = self.drop(x)
        if self.fc is None: self.fc = nn.Linear(x.size(1), 3).to(x.device)
        return self.fc(x)




# ====== 2024-2025 Published Models ======

class TCN2024(nn.Module):
    """Temporal Convolutional Network (TCN) - widely used in 2024 EEG studies.
    Dilated causal convolutions with residual connections.
    Reference: Bai et al., 2018; adapted for EEG regression in multiple 2024 works."""
    def __init__(self, input_size=5, output_size=3, num_channels=[32,32,32], kernel_size=15, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                                     padding=(kernel_size-1)*dilation//2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (B, 100, 145) -> (B, 5, 29*100) for Conv1d
        B, T, F = x.shape
        x = x.reshape(B, T, 29, 5).permute(0, 3, 2, 1).reshape(B, 5, 29*T)
        x = self.conv_stack(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class InceptionTime2024(nn.Module):
    """InceptionTime - multi-scale 1D convolutions for time series.
    Widely adopted in 2023-2024 EEG classification/regression papers.
    Reference: Ismail Fawaz et al., 2020; 2024 EEG adaptations."""
    def __init__(self, input_size=5, output_size=3, n_filters=16, depth=3, dropout=0.3):
        super().__init__()
        self.inception_blocks = nn.ModuleList()
        in_ch = input_size
        for _ in range(depth):
            block = nn.ModuleDict({
                'conv_40': nn.Conv1d(in_ch, n_filters, 40, padding=19),
                'conv_20': nn.Conv1d(in_ch, n_filters, 20, padding=9),
                'conv_10': nn.Conv1d(in_ch, n_filters, 10, padding=4),
                'maxpool': nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                          nn.Conv1d(in_ch, n_filters, 1)),
                'bn': nn.BatchNorm1d(n_filters * 4),
                'relu': nn.ReLU(),
                'dropout': nn.Dropout(dropout),
            })
            self.inception_blocks.append(block)
            in_ch = n_filters * 4
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, output_size)

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B, T, 29, 5).permute(0, 3, 2, 1).reshape(B, 5, 29*T)
        for block in self.inception_blocks:
            x1 = block['conv_40'](x)
            x2 = block['conv_20'](x)
            x3 = block['conv_10'](x)
            x4 = block['maxpool'](x)
            min_len = min(x1.size(-1), x2.size(-1), x3.size(-1), x4.size(-1))
            x1, x2, x3, x4 = x1[:,:,:min_len], x2[:,:,:min_len], x3[:,:,:min_len], x4[:,:,:min_len]
            x = torch.cat([x1, x2, x3, x4], dim=1)
            x = block['relu'](block['bn'](x))
            x = block['dropout'](x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class GatedTCN2025(nn.Module):
    """Gated TCN with channel attention - 2025 architecture for biosignal regression.
    Combines dilated TCN with gating mechanism and SE attention.
    Reference: Adapted from recent 2024-2025 EEG regression works."""
    def __init__(self, input_size=5, output_size=3, hidden=32, n_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, hidden, 1)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(nn.ModuleDict({
                'filter_conv': nn.Conv1d(hidden, hidden, 5, dilation=dilation,
                                          padding=2*dilation),
                'gate_conv': nn.Conv1d(hidden, hidden, 5, dilation=dilation,
                                        padding=2*dilation),
                'filter_bn': nn.BatchNorm1d(hidden),
                'gate_bn': nn.BatchNorm1d(hidden),
                'se': nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                     nn.Conv1d(hidden, hidden//4, 1), nn.ReLU(),
                                     nn.Conv1d(hidden//4, hidden, 1), nn.Sigmoid()),
                'dropout': nn.Dropout(dropout),
            }))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B, T, 29, 5).permute(0, 3, 2, 1).reshape(B, 5, 29*T)
        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            filt = torch.tanh(layer['filter_bn'](layer['filter_conv'](x)))
            gate = torch.sigmoid(layer['gate_bn'](layer['gate_conv'](x)))
            x = filt * gate
            se_weight = layer['se'](x)
            x = x * se_weight
            x = layer['dropout'](x)
            x = x + residual
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


print('TCN + InceptionTime + GatedTCN ready')


# ====== MAIN ======
if __name__ == '__main__':
    import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('Loading data...')
    X, y, ys = load_data('h5_files', '总表.xlsx')  # TODO: update path to your metadata file
    print(f'X: {X.shape}, y: {y.shape}')

    models_to_run = [
        ('FBCNet', FBCNet),
        ('TSCeption', TSCeption),
        ('TCN (2024)', TCN2024),
        ('InceptionTime (2024)', InceptionTime2024),
        ('GatedTCN (2025)', GatedTCN2025),
    ]
    all_sums = {}
    for name, cls in models_to_run:
        print(f'\n=== {name} ===')
        all_sums[name] = run_cv(cls, X, y, ys, TARGET_COLS)

    print('\n=== COMPARISON TABLE ===')
    print('Model                    | ADL R          | FMA R          | FMA-UE R')
    print('-' * 75)
    ref = {'ADL':{'mean_r':0.68,'std_r':0.07},'FMA':{'mean_r':0.81,'std_r':0.03},'FMA-UE':{'mean_r':0.80,'std_r':0.02}}
    for name, s in list(all_sums.items()) + [('ESTAF-SMS (ref)', ref)]:
        print(f'{name:<25}| {s["ADL"]["mean_r"]:.2f}+/-{s["ADL"]["std_r"]:.2f}        | {s["FMA"]["mean_r"]:.2f}+/-{s["FMA"]["std_r"]:.2f}        | {s["FMA-UE"]["mean_r"]:.2f}+/-{s["FMA-UE"]["std_r"]:.2f}')
    print('\nDone!')
