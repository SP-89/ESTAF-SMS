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
    import os; os.chdir(ros.path.join(os.path.dirname(__file__), "model"))
    print('Loading data...')
    X, y, ys = load_data('h5_files', '\u603b\u8868.xlsx')
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


print('FBCNet + TSCeption ready')

# ====== MAIN ======
if __name__ == '__main__':
    import os; os.chdir(ros.path.join(os.path.dirname(__file__), "model"))
    print('Loading data...')
    X, y, ys = load_data('h5_files', 'metadata.xlsx')
    print(f'X: {X.shape}, y: {y.shape}')

    print('\n=== FBCNet (Mane et al., 2021) ===')
    fbc_sum = run_cv(FBCNet, X, y, ys, TARGET_COLS)

    print('\n=== TSCeption (Ding et al., 2022) ===')
    tsc_sum = run_cv(TSCeption, X, y, ys, TARGET_COLS)

    print('\n=== COMPARISON TABLE ===')
    print('Model               | ADL R          | FMA R          | FMA-UE R')
    print('-' * 70)
    for name, s in [('FBCNet', fbc_sum), ('TSCeption', tsc_sum), ('ESTAF-SMS (ref)', {'ADL':{'mean_r':0.68,'std_r':0.07},'FMA':{'mean_r':0.81,'std_r':0.03},'FMA-UE':{'mean_r':0.80,'std_r':0.02}})]:
        print(f'{name:<20}| {s["ADL"]["mean_r"]:.2f}+/-{s["ADL"]["std_r"]:.2f}        | {s["FMA"]["mean_r"]:.2f}+/-{s["FMA"]["std_r"]:.2f}        | {s["FMA-UE"]["mean_r"]:.2f}+/-{s["FMA-UE"]["std_r"]:.2f}')
    print('\nDone!')
