import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.signal import welch
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def main(input_dir, output_dir, sampling_rate=1000,
                       target_segments=100, window_sec=2, overlap=0.5):
    """
    PSD Visualization Only - No H5 File Generation
    """
    # Create visualization output directory
    vis_dir = os.path.join(output_dir, "psd_visualizations")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # Get MAT files
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))
    if not mat_files:
        print(f"No MAT files found in directory: {input_dir}!")
        return

    print(f"Found {len(mat_files)} MAT files")
    print("Starting PSD visualization...")

    # Process each file for visualization
    for file_path in tqdm(mat_files, desc="PSD Visualization"):
        try:
            # Load data
            eeg_data = load_and_validate_data(file_path)
            if eeg_data is None:
                continue

            # Extract PSD features for visualization
            psd_features = extract_psd_features_only(
                eeg_data, sampling_rate, target_segments,
                window_sec, overlap
            )

            if psd_features is not None:
                sample_name = os.path.splitext(os.path.basename(file_path))[0]
                print(f"Visualizing: {sample_name}")

                # Generate visualization
                create_comprehensive_psd_visualization(
                    psd_features, sample_name, vis_dir, sampling_rate
                )

        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
            continue

    print(f"\nPSD visualization completed! All images saved in: {vis_dir}")


def load_and_validate_data(file_path):
    """
    Load and validate EEG data
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

                # Simple data validation
                data_std = np.std(eeg_data)
                if data_std < 1e-6 or data_std > 1e6 or np.any(np.isnan(eeg_data)):
                    print(f"Data validation failed: {os.path.basename(file_path)}")
                    return None

                return eeg_data

        print(f"Warning: No 29-channel data found in {os.path.basename(file_path)}")
        return None

    except Exception as e:
        print(f"Failed to load file {os.path.basename(file_path)}: {str(e)}")
        return None


def extract_psd_features_only(eeg_data, sampling_rate, target_segments,
                              window_sec, overlap):
    """
    Extract PSD features for visualization
    """
    n_channels, total_points = eeg_data.shape
    window_points = int(window_sec * sampling_rate)
    step_points = int(window_points * (1 - overlap))

    n_windows = max(0, (total_points - window_points) // step_points + 1)
    if n_windows < target_segments:
        print(f"Warning: Only {n_windows} windows, less than target {target_segments}")
        return None

    if n_windows > target_segments:
        skip = (n_windows - target_segments) // 4
        window_indices = np.linspace(skip, n_windows - skip - 1, target_segments, dtype=int)
    else:
        window_indices = np.arange(n_windows)

    # Initialize PSD feature array
    n_freq_points = 90  # 0.5-45Hz range
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
            channel_data = channel_data - np.mean(channel_data)  # Remove DC component

            # Extract PSD
            psd_values = extract_single_channel_psd(channel_data, sampling_rate, window_points)
            psd_features[seg_idx, ch_idx, :] = psd_values

    return psd_features


def extract_single_channel_psd(channel_data, sampling_rate, data_length):
    """
    Extract PSD for single channel
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

        # Extract PSD values in 0.5-45Hz range
        total_mask = (freqs >= 0.5) & (freqs <= 45)
        psd_values = psd[total_mask]

        # Ensure exactly 90 frequency points
        if len(psd_values) > 90:
            psd_values = psd_values[:90]
        elif len(psd_values) < 90:
            padded_psd = np.zeros(90, dtype=np.float32)
            padded_psd[:len(psd_values)] = psd_values
            psd_values = padded_psd

        return psd_values

    except Exception as e:
        print(f"PSD calculation failed: {str(e)}")
        return np.zeros(90, dtype=np.float32)


def create_comprehensive_psd_visualization(psd_features, sample_name, output_dir, sampling_rate=1000):
    """
    Create comprehensive PSD visualization
    """
    n_segments, n_channels, n_freqs = psd_features.shape
    freqs = np.linspace(0.5, 45, n_freqs)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create main figure
    fig = plt.figure(figsize=(20, 16))

    # 1. Average PSD across all channels (top-left)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    mean_psd_all = np.mean(psd_features, axis=(0, 1))
    std_psd_all = np.std(psd_features, axis=(0, 1))

    plt.plot(freqs, mean_psd_all, 'b-', linewidth=2, label='Mean PSD')
    plt.fill_between(freqs, mean_psd_all - std_psd_all, mean_psd_all + std_psd_all,
                     alpha=0.3, label='±1 STD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Average PSD Across All Channels - {sample_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Frequency band power distribution (top-right)
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    bands = {
        'Delta (0.5-4Hz)': (0.5, 4),
        'Theta (4-8Hz)': (4, 8),
        'Alpha (8-13Hz)': (8, 13),
        'Beta (13-30Hz)': (13, 30),
        'Gamma (30-45Hz)': (30, 45)
    }

    band_powers = []
    band_names = []
    for band_name, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = np.mean(mean_psd_all[band_mask])
        band_powers.append(band_power)
        band_names.append(band_name.split(' ')[0])  # Only take band name

    colors = plt.cm.Set3(np.linspace(0, 1, len(bands)))
    bars = plt.bar(band_names, band_powers, color=colors, alpha=0.8)
    plt.xticks(rotation=45)
    plt.ylabel('Average Power')
    plt.title('Power Distribution by Frequency Band')

    # Add values on bars
    for bar, value in zip(bars, band_powers):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{value:.2e}', ha='center', va='bottom', fontsize=9)

    # 3. Channel PSD heatmap (middle-left)
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    channel_avg_psd = np.mean(psd_features, axis=0)

    im = plt.imshow(channel_avg_psd, aspect='auto', cmap='viridis',
                    extent=[freqs[0], freqs[-1], n_channels, 1])
    plt.colorbar(im, label='Power', shrink=0.8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channel Number')
    plt.title('PSD Heatmap by Channel')
    plt.yticks(range(1, n_channels + 1, 4))  # Show label every 4 channels

    # 4. PSD variation across time segments (middle-right)
    ax4 = plt.subplot2grid((3, 3), (1, 2))
    n_segments_to_plot = min(6, n_segments)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_segments_to_plot))

    for i in range(n_segments_to_plot):
        segment_avg_psd = np.mean(psd_features[i], axis=0)
        plt.plot(freqs, segment_avg_psd, alpha=0.7, color=colors[i],
                 label=f'Segment {i + 1}', linewidth=1.5)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title(f'PSD Variation Across First {n_segments_to_plot} Segments')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 5. Channel power variability (bottom-left)
    ax5 = plt.subplot2grid((3, 3), (2, 0))
    channel_power_variability = np.std(channel_avg_psd, axis=1)

    plt.bar(range(1, n_channels + 1), channel_power_variability,
            alpha=0.7, color='steelblue')
    plt.xlabel('Channel Number')
    plt.ylabel('Power Standard Deviation')
    plt.title('Power Variability Across Channels')
    plt.grid(True, alpha=0.3)

    # 6. Frequency band power comparison boxplot (bottom-middle)
    ax6 = plt.subplot2grid((3, 3), (2, 1))
    band_data = []
    band_labels = []

    for band_name, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_values = channel_avg_psd[:, band_mask].flatten()
        band_data.append(band_values)
        band_labels.append(band_name.split(' ')[0])

    box_plot = plt.boxplot(band_data, labels=band_labels, patch_artist=True)
    # Set boxplot colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(bands)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(rotation=45)
    plt.ylabel('Power')
    plt.title('Power Distribution by Frequency Band')
    plt.grid(True, alpha=0.3)

    # 7. Total power distribution (bottom-right)
    ax7 = plt.subplot2grid((3, 3), (2, 2))
    total_power = np.sum(channel_avg_psd, axis=1)

    plt.bar(range(1, n_channels + 1), total_power, alpha=0.7, color='coral')
    plt.xlabel('Channel Number')
    plt.ylabel('Total Power')
    plt.title('Total Power by Channel')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample_name}_comprehensive_psd.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create individual channel PSD plots
    create_individual_channel_plots(psd_features, sample_name, output_dir, freqs)

    print(f"  - Generated: {sample_name}_comprehensive_psd.png")
    print(f"  - Generated: {sample_name}_channel_details.png")


def create_individual_channel_plots(psd_features, sample_name, output_dir, freqs):
    """
    Create detailed PSD plots for individual channels
    """
    n_segments, n_channels, n_freqs = psd_features.shape

    # Select representative channels (first 12 channels)
    channels_to_plot = min(12, n_channels)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()

    # Define frequency bands and colors
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
             'Beta': (13, 30), 'Gamma': (30, 45)}
    band_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i in range(channels_to_plot):
        channel_psd = psd_features[:, i, :]

        # Calculate statistics
        mean_psd = np.mean(channel_psd, axis=0)
        std_psd = np.std(channel_psd, axis=0)

        # Plot PSD for this channel
        axes[i].plot(freqs, mean_psd, 'k-', linewidth=2, label='Mean PSD')
        axes[i].fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd,
                             alpha=0.3, color='gray', label='±1 STD')

        # Add frequency band background colors
        for (band_name, (low, high)), color in zip(bands.items(), band_colors):
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                axes[i].axvspan(low, high, alpha=0.2, color=color, label=band_name)

        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('PSD')
        axes[i].set_title(f'Channel {i + 1} PSD Analysis')
        axes[i].grid(True, alpha=0.3)

        # Show legend only on the first plot
        if i == 0:
            axes[i].legend(fontsize=8, loc='upper right')

    # Hide extra subplots
    for i in range(channels_to_plot, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Individual Channel PSD Analysis - {sample_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample_name}_channel_details.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# Usage example
if __name__ == "__main__":
    input_directory = "mat_files"  # Change to your MAT file directory
    output_directory = "psd_visualizations_output"

    print("=" * 60)
    print("EEG PSD Visualization Program")
    print("=" * 60)
    print("Configuration:")
    print("  - Input directory:", input_directory)
    print("  - Output directory:", output_directory)
    print("  - Sampling rate: 1000 Hz")
    print("  - Window length: 2 seconds")
    print("  - Overlap: 50%")
    print("  - Frequency range: 0.5-45Hz")
    print("=" * 60)

    main(
        input_dir=input_directory,
        output_dir=output_directory,
        sampling_rate=1000,
        target_segments=100,
        window_sec=2,
        overlap=0.5
    )

    print("=" * 60)
    print("PSD Visualization Completed!")
    print(f"Output directory: {os.path.abspath(output_directory)}/psd_visualizations/")
    print("=" * 60)