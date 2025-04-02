import os
from glob import glob  # Kept in case it's used elsewhere

def get_image_list(data_root, split):
    filelist = []
    file_list_path = os.path.join('filelists', f'{split}.txt')
    with open(file_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ' ' in line:
                line = line.split()[0]
            filelist.append(os.path.join(data_root, line))
    return filelist


class HParams:
    def __init__(self, **kwargs):
        self.data = {}
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError(f"'HParams' object has no attribute {key}")
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value

    def values(self):
        """Return the dictionary of hyperparameters."""
        return self.data


# Default hyperparameters
hparams = HParams(
    # Mel-spectrogram parameters
    num_mels=80,              # Number of mel-spectrogram channels and conditioning dimensions
    fmin=55,                  # Lower boundary frequency for mel filters (used in assertions)
    fmax=7600,                # Upper boundary frequency for mel filters (used in assertions)
    mel_fmin=55,              # Lower frequency bound for mel filters (used in librosa.filters.mel)
    mel_fmax=7600,            # Upper frequency bound for mel filters (used in librosa.filters.mel)

    # Audio preprocessing parameters
    rescale=True,             # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,        # Maximum rescaling value

    # Use LWS for STFT and phase reconstruction (useful with some vocoders)
    use_lws=False,

    n_fft=800,                # FFT window size; data is padded with zeros if needed
    hop_size=200,             # Hop size in samples (For 16000 Hz, 200 samples ≈ 12.5 ms)
    win_size=800,             # Window size in samples; if None, win_size = n_fft
    sample_rate=16000,        # Audio sampling rate in Hz

    frame_shift_ms=None,      # Alternative to hop_size (recommended around 12.5 ms)

    # Spectrogram normalization/scaling options
    signal_normalization=True,
    allow_clipping_in_normalization=True,  # Relevant for mel normalization
    symmetric_mels=True,  # Scale data symmetrically about zero
    max_abs_value=4.,     # Maximum absolute value to avoid gradient explosion

    # Pre-emphasis filtering parameters
    preemphasize=True,
    preemphasis=0.97,

    # Clipping limits
    min_level_db=-100,
    ref_level_db=20,

    ###################### Training Parameters #####################
    img_size=96,
    fps=25,
    
    batch_size=16,
    initial_learning_rate=1e-4,
    nepochs=200000000000000000,  # Very high value; stop training manually based on evaluation loss
    num_workers=16,
    checkpoint_interval=3000,
    eval_interval=3000,
    save_optimizer_state=True,

    # SyncNet-related parameters (for faster convergence)
    syncnet_wt=0.0,           # Initially set to 0.0; will be updated to around 0.03 later during training
    syncnet_batch_size=64,
    syncnet_lr=1e-4,
    syncnet_eval_interval=10000,
    syncnet_checkpoint_interval=10000,

    # Discriminator parameters for adversarial training
    disc_wt=0.07,
    disc_initial_learning_rate=1e-4,
)


def hparams_debug_string():
    values = hparams.values()
    hp_lines = [f"  {name}: {values[name]}" for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp_lines)
