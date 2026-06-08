import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.signal import welch, hilbert
import os
from tqdm import tqdm
import glob
import h5py
import warnings
from scipy.stats import kurtosis, skew

warnings.filterwarnings('ignore')


def full_feature_simple(input_dir, output_dir, sampling_rate=1000,
                        target_segments=100, window_sec=2, overlap=0.5):
    """
    只提取PSD特征 - 输出形状 (100, 29, 90)
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取MAT文件
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))
    if not mat_files:
        print(f"在目录 {input_dir} 中未找到MAT文件!")
        return {}

    print(f"找到 {len(mat_files)} 个MAT文件")
    print(f"特征配置：2秒窗口，{overlap * 100}%重叠，只提取PSD特征，形状(100, 29, 90)")

    processed_count = 0

    for file_path in tqdm(mat_files, desc="PSD特征处理"):
        try:
            # 加载并验证数据
            eeg_data = load_mat_data(file_path)
            if eeg_data is None:
                continue

            # PSD特征提取
            psd_features = extract_psd_features(
                eeg_data, sampling_rate, target_segments,
                window_sec, overlap
            )

            if psd_features is not None:
                # 保存为H5
                save_psd_features_h5(psd_features, file_path, output_dir,
                                     window_sec, overlap, sampling_rate)

                processed_count += 1
                print(f"已完成: {os.path.basename(file_path)} - PSD特征形状: {psd_features.shape}")

        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
            continue

    print(f"\n处理完成! 成功处理 {processed_count}/{len(mat_files)} 个文件")
    return processed_count


def load_mat_data(file_path):
    """
    加载并验证EEG数据
    """
    try:
        mat_data = sio.loadmat(file_path)
        for key in mat_data.keys():
            if not key.startswith('__'):
                data = mat_data[key]
                if data.shape[0] == 29:
                    eeg_data = data.astype(np.float32)
                elif data.shape[1] == 29:
                    eeg_data = data.T.astype(np.float32)
                else:
                    continue

                # 数据验证
                if validate_eeg_data(eeg_data, os.path.basename(file_path)):
                    return eeg_data
                else:
                    print(f"数据验证失败: {os.path.basename(file_path)}")
                    return None

        print(f"警告: {os.path.basename(file_path)} 中未找到29通道数据")
        return None

    except Exception as e:
        print(f"加载文件 {os.path.basename(file_path)} 失败: {str(e)}")
        return None


def validate_eeg_data(eeg_data, filename):
    """
    验证EEG数据的合理性
    """
    # 检查数据范围
    data_min, data_max = np.min(eeg_data), np.max(eeg_data)
    data_std = np.std(eeg_data)
    data_mean = np.mean(eeg_data)

    print(
        f"{filename} - 数据统计: 最小值: {data_min:.4f}, 最大值: {data_max:.4f}, 均值: {data_mean:.4f}, 标准差: {data_std:.4f}")

    # 检查数据是否合理
    if data_std < 1e-6:  # 数据过于平坦
        print(f"警告: {filename} 数据标准差过小，可能已过度归一化")
        return False
    elif data_std > 1e6:  # 数据范围过大
        print(f"警告: {filename} 数据标准差过大")
        return False
    elif np.isnan(data_min) or np.isnan(data_max):  # 包含NaN值
        print(f"警告: {filename} 包含NaN值")
        return False

    return True


def extract_psd_features(eeg_data, sampling_rate, target_segments,
                              window_sec, overlap):
    """
    只提取PSD特征 - 输出形状 (100, 29, 90)
    """
    n_channels, total_points = eeg_data.shape
    window_points = int(window_sec * sampling_rate)
    step_points = int(window_points * (1 - overlap))

    n_windows = max(0, (total_points - window_points) // step_points + 1)
    if n_windows < target_segments:
        print(f"警告: 只有 {n_windows} 个窗口，少于目标 {target_segments}")
        return None

    if n_windows > target_segments:
        skip = (n_windows - target_segments) // 4
        window_indices = np.linspace(skip, n_windows - skip - 1, target_segments, dtype=int)
    else:
        window_indices = np.arange(n_windows)

    # 预计算频率范围，确定total_mask的长度
    test_data = eeg_data[0, :window_points]
    test_data = test_data - np.mean(test_data)

    try:
        freqs, psd = welch(test_data,
                           fs=sampling_rate,
                           window=signal.windows.hann(window_points),
                           nperseg=window_points,
                           noverlap=window_points // 2,
                           nfft=window_points,
                           scaling='density',
                           average='mean',
                           return_onesided=True)

        total_mask = (freqs >= 0.5) & (freqs <= 45)
        n_freq_points = np.sum(total_mask)
        print(f"PSD频率点数量: {n_freq_points} (0.5-45Hz)")

    except Exception as e:
        print(f"预计算PSD频率点失败: {str(e)}")
        n_freq_points = 90  # 默认值

    # 初始化PSD特征数组: (窗口数, 通道数, 频率点数)
    psd_features = np.zeros((len(window_indices), n_channels, n_freq_points), dtype=np.float32)

    for seg_idx, win_idx in enumerate(window_indices):
        start = win_idx * step_points
        end = start + window_points
        if end > total_points:
            start = total_points - window_points
            end = total_points

        window_data = eeg_data[:, start:end]

        for ch_idx in range(n_channels):
            channel_data = window_data[ch_idx, :]

            # 数据预处理
            channel_data = preprocess_channel_data(channel_data)

            # 提取PSD特征
            psd_values = extract_psd_only(channel_data, sampling_rate, window_points, ch_idx)

            if psd_values is not None and len(psd_values) == n_freq_points:
                psd_features[seg_idx, ch_idx, :] = psd_values
            else:
                print(f"警告: 通道 {ch_idx} PSD特征长度不匹配")

    return psd_features


def preprocess_channel_data(channel_data):
    """
    通道数据预处理
    """
    # 移除直流分量
    channel_data = channel_data - np.mean(channel_data)
    # 检查并处理异常值
    data_std = np.std(channel_data)
    if data_std < 1e-6:
        # 如果标准差太小，添加微小噪声避免除零错误
        channel_data = channel_data + np.random.normal(0, 1e-6, len(channel_data))

    return channel_data


def extract_psd_only(channel_data, sampling_rate, data_length, ch_idx=0):
    """
    只提取PSD特征
    """
    try:
        freqs, psd = welch(channel_data,
                           fs=sampling_rate,
                           window=signal.windows.hann(data_length),
                           nperseg=data_length,
                           noverlap=data_length // 2,
                           nfft=data_length,
                           scaling='density',
                           average='mean',
                           return_onesided=True)

        # 提取0.5-45Hz范围内的PSD值
        total_mask = (freqs >= 0.5) & (freqs <= 45)
        psd_values = psd[total_mask]

        # 确保有90个频率点
        if len(psd_values) > 90:
            psd_values = psd_values[:90]
        elif len(psd_values) < 90:
            # 如果点数不足，用零填充
            padded_psd = np.zeros(90, dtype=np.float32)
            padded_psd[:len(psd_values)] = psd_values
            psd_values = padded_psd

        return psd_values

    except Exception as e:
        print(f"通道 {ch_idx} PSD计算失败: {str(e)}")
        # 返回零值特征
        return np.zeros(90, dtype=np.float32)


def validate_psd(psd, freqs, ch_idx):
    """
    验证PSD结果的合理性
    """
    # 检查PSD值范围
    psd_min, psd_max = np.min(psd), np.max(psd)
    psd_mean = np.mean(psd)

    # 正常EEG PSD应该在合理范围内
    if psd_max > 1e6 or (psd_max < 1e-10 and psd_min < 1e-10):
        print(f"通道 {ch_idx} PSD范围异常: {psd_min:.2e} to {psd_max:.2e}")
        return False

    # 检查是否有过多零值或异常值
    zero_count = np.sum(psd < 1e-15)
    if zero_count > len(psd) * 0.5:  # 如果超过50%的值接近零
        print(f"通道 {ch_idx} 过多接近零的PSD值: {zero_count}/{len(psd)}")
        return False

    return True


def save_psd_features_h5(psd_features, input_file_path, output_dir,
                         window_sec, overlap, sampling_rate):
    """
    保存PSD特征到H5文件
    """
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.h5")

    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('psd_features', data=psd_features, compression='gzip', compression_opts=4)

        h5f.attrs['sampling_rate'] = sampling_rate
        h5f.attrs['window_seconds'] = window_sec
        h5f.attrs['overlap_ratio'] = overlap
        h5f.attrs['n_segments'] = psd_features.shape[0]
        h5f.attrs['n_channels'] = psd_features.shape[1]
        h5f.attrs['n_freq_points'] = psd_features.shape[2]
        h5f.attrs['original_filename'] = os.path.basename(input_file_path)

        current_time = np.datetime64('now').astype(str)
        h5f.attrs['processing_date'] = np.bytes_(current_time.encode('utf-8'))

        h5f.attrs['feature_type'] = np.bytes_('PSD_only'.encode('utf-8'))
        h5f.attrs['frequency_range'] = np.bytes_('0.5-45Hz'.encode('utf-8'))
        h5f.attrs['processing_method'] = np.bytes_('psd_features_v1.0'.encode('utf-8'))

        # 保存PSD配置信息
        h5f.attrs['psd_window_type'] = np.bytes_('hann'.encode('utf-8'))
        h5f.attrs['psd_nperseg'] = int(window_sec * sampling_rate)
        h5f.attrs['psd_noverlap'] = int(window_sec * sampling_rate) // 2
        h5f.attrs['psd_nfft'] = int(window_sec * sampling_rate)
        h5f.attrs['psd_scaling'] = np.bytes_('density'.encode('utf-8'))
        h5f.attrs['psd_averaging'] = np.bytes_('mean'.encode('utf-8'))
        h5f.attrs['psd_sides'] = np.bytes_('onesided'.encode('utf-8'))

    return output_path


def process_single_file_psd_only(file_path, output_dir, sampling_rate,
                                 target_segments, window_sec, overlap):
    """单个文件的并行处理函数 - 只提取PSD"""
    try:
        # 加载数据
        eeg_data = load_mat_data(file_path)
        if eeg_data is None:
            return 0

        # PSD特征提取
        psd_features = extract_psd_features(
            eeg_data, sampling_rate, target_segments,
            window_sec, overlap
        )

        if psd_features is not None:
            # 保存
            save_psd_features_h5(psd_features, file_path, output_dir,
                                 window_sec, overlap, sampling_rate)
            return 1
        else:
            return 0

    except Exception as e:
        print(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)[:100]}...")
        return 0


# 使用示例
if __name__ == "__main__":
    input_directory = "mat_files"
    output_directory = "h5_files"

    params = {
        'sampling_rate': 1000,
        'target_segments': 100,
        'window_sec': 2,
        'overlap': 0.5
    }

    print("=" * 60)
    print("EEG PSD特征提取程序")
    print("=" * 60)
    print("PSD配置:")
    print("  - 窗口: 汉宁窗 (hann)")
    print("  - 窗口长度: 2秒 × 1000Hz = 2000点")
    print("  - 重叠: 50% (1000点)")
    print("  - FFT点数: 2000点")
    print("  - 谱类型: 单边功率谱密度")
    print("特征配置:")
    print("  - 只提取PSD特征")
    print("  - 频率范围: 0.5-45Hz")
    print("  - 输出形状: (100, 29, 90)")
    print("=" * 60)

    processed_count = full_feature_simple(
        input_dir=input_directory,
        output_dir=output_directory,
        **params
    )

    print(f"\n" + "=" * 60)
    print(f"最终统计: 成功处理 {processed_count} 个文件")
    print(f"输出目录: {os.path.abspath(output_directory)}")
    print("特征详情:")
    print("  - 只保留PSD特征，形状为 (100个时间窗口, 29个通道, 90个频率点)")
    print("  - 频率范围: 0.5-45Hz，共90个频率点")
    print("=" * 60)