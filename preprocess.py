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
    Extract onlyPSD -  (100, 29, 90)
    """

    # Createdirectory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # MATFile
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))
    if not mat_files:
        print(f"directory {input_dir} FoundMATFile!")
        return {}

    print(f"Found {len(mat_files)} MATFile")
    print(f"Configuration：2，{overlap * 100}%Overlap，Extract onlyPSD，(100, 29, 90)")

    processed_count = 0

    for file_path in tqdm(mat_files, desc="PSDProcessing"):
        try:
            # Load and validate data
            eeg_data = load_mat_data(file_path)
            if eeg_data is None:
                continue

            # PSD
            psd_features = extract_psd_features(
                eeg_data, sampling_rate, target_segments,
                window_sec, overlap
            )

            if psd_features is not None:
                # H5
                save_psd_features_h5(psd_features, file_path, output_dir,
                                     window_sec, overlap, sampling_rate)

                processed_count += 1
                print(f"Done: {os.path.basename(file_path)} - PSDFeature shape: {psd_features.shape}")

        except Exception as e:
            print(f"ProcessingFile {os.path.basename(file_path)} Error:: {str(e)}")
            continue

    print(f"\nProcessing complete! successfullyProcessing {processed_count}/{len(mat_files)} File")
    return processed_count


def load_mat_data(file_path):
    """
    LoadEEG
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


                if validate_eeg_data(eeg_data, os.path.basename(file_path)):
                    return eeg_data
                else:
                    print(f"Data validation failed: {os.path.basename(file_path)}")
                    return None

        print(f"Warning: {os.path.basename(file_path)} Found29")
        return None

    except Exception as e:
        print(f"LoadFile {os.path.basename(file_path)} : {str(e)}")
        return None


def validate_eeg_data(eeg_data, filename):
    """
    EEG
    """

    data_min, data_max = np.min(eeg_data), np.max(eeg_data)
    data_std = np.std(eeg_data)
    data_mean = np.mean(eeg_data)

    print(
        f"{filename} - : : {data_min:.4f}, : {data_max:.4f}, : {data_mean:.4f}, : {data_std:.4f}")


    if data_std < 1e-6:
        print(f"Warning: {filename} ，")
        return False
    elif data_std > 1e6:
        print(f"Warning: {filename} ")
        return False
    elif np.isnan(data_min) or np.isnan(data_max):  # NaN
        print(f"Warning: {filename} NaN")
        return False

    return True


def extract_psd_features(eeg_data, sampling_rate, target_segments,
                              window_sec, overlap):
    """
    Extract onlyPSD -  (100, 29, 90)
    """
    n_channels, total_points = eeg_data.shape
    window_points = int(window_sec * sampling_rate)
    step_points = int(window_points * (1 - overlap))

    n_windows = max(0, (total_points - window_points) // step_points + 1)
    if n_windows < target_segments:
        print(f"Warning:  {n_windows} ，target {target_segments}")
        return None

    if n_windows > target_segments:
        skip = (n_windows - target_segments) // 4
        window_indices = np.linspace(skip, n_windows - skip - 1, target_segments, dtype=int)
    else:
        window_indices = np.arange(n_windows)

    # Frequency range，total_mask
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
        print(f"PSD: {n_freq_points} (0.5-45Hz)")

    except Exception as e:
        print(f"PSD: {str(e)}")
        n_freq_points = 90

    # PSD: (, , )
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


            channel_data = preprocess_channel_data(channel_data)

            # Extract PSD features
            psd_values = extract_psd_only(channel_data, sampling_rate, window_points, ch_idx)

            if psd_values is not None and len(psd_values) == n_freq_points:
                psd_features[seg_idx, ch_idx, :] = psd_values
            else:
                print(f"Warning:  {ch_idx} PSD")

    return psd_features


def preprocess_channel_data(channel_data):
    """
    Processing
    """

    channel_data = channel_data - np.mean(channel_data)

    data_std = np.std(channel_data)
    if data_std < 1e-6:
        # ，
        channel_data = channel_data + np.random.normal(0, 1e-6, len(channel_data))

    return channel_data


def extract_psd_only(channel_data, sampling_rate, data_length, ch_idx=0):
    """
    Extract onlyPSD
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

        # 0.5-45HzPSD
        total_mask = (freqs >= 0.5) & (freqs <= 45)
        psd_values = psd[total_mask]

        # 90
        if len(psd_values) > 90:
            psd_values = psd_values[:90]
        elif len(psd_values) < 90:
            # ，
            padded_psd = np.zeros(90, dtype=np.float32)
            padded_psd[:len(psd_values)] = psd_values
            psd_values = padded_psd

        return psd_values

    except Exception as e:
        print(f" {ch_idx} PSD calculation failed: {str(e)}")

        return np.zeros(90, dtype=np.float32)


def validate_psd(psd, freqs, ch_idx):
    """
    PSD
    """
    # PSD
    psd_min, psd_max = np.min(psd), np.max(psd)
    psd_mean = np.mean(psd)

    # EEG PSD
    if psd_max > 1e6 or (psd_max < 1e-10 and psd_min < 1e-10):
        print(f" {ch_idx} PSD: {psd_min:.2e} to {psd_max:.2e}")
        return False


    zero_count = np.sum(psd < 1e-15)
    if zero_count > len(psd) * 0.5:  # 50%
        print(f" {ch_idx} PSD: {zero_count}/{len(psd)}")
        return False

    return True


def save_psd_features_h5(psd_features, input_file_path, output_dir,
                         window_sec, overlap, sampling_rate):
    """
    savedPSDH5File
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

        # PSDConfiguration
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
    """FileProcessing - Extract onlyPSD"""
    try:
        # loading data
        eeg_data = load_mat_data(file_path)
        if eeg_data is None:
            return 0

        # PSD
        psd_features = extract_psd_features(
            eeg_data, sampling_rate, target_segments,
            window_sec, overlap
        )

        if psd_features is not None:

            save_psd_features_h5(psd_features, file_path, output_dir,
                                 window_sec, overlap, sampling_rate)
            return 1
        else:
            return 0

    except Exception as e:
        print(f"ProcessingFile {os.path.basename(file_path)} Error:: {str(e)[:100]}...")
        return 0


# Usage
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
    print("EEG PSDExtractProgram")
    print("=" * 60)
    print("PSDConfiguration:")
    print("  - :  (hann)")
    print("  - Window length: 2 × 1000Hz = 2000")
    print("  - Overlap: 50% (1000)")
    print("  - FFT: 2000")
    print("  - : ")
    print("Configuration:")
    print("  - Extract onlyPSD")
    print("  - Frequency range: 0.5-45Hz")
    print("  - : (100, 29, 90)")
    print("=" * 60)

    processed_count = full_feature_simple(
        input_dir=input_directory,
        output_dir=output_directory,
        **params
    )

    print(f"\n" + "=" * 60)
    print(f": successfullyProcessing {processed_count} File")
    print(f"directory: {os.path.abspath(output_directory)}")
    print(":")
    print("  - PSD， (100, 29, 90)")
    print("  - Frequency range: 0.5-45Hz，90")
    print("=" * 60)