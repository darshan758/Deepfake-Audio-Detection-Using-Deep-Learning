import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path


class AudioDeepfakeDataset(Dataset):
    """
    Custom Dataset for Deepfake Audio Detection.
    Loads .wav files, extracts MFCC features, pads them to a fixed length,
    and returns tensors for model training.
    """

    def __init__(self, file_list, labels, n_mfcc: int = 40, max_length: int = 800):
        """
        Args:
            file_list (list): List of paths to audio files.
            labels (list): List of labels (0 = real, 1 = fake).
            n_mfcc (int): Number of MFCC features to extract.
            max_length (int): Fixed length (time frames) for all samples.
        """
        assert len(file_list) == len(labels), "File list and labels length mismatch!"
        self.file_list = file_list
        self.labels = labels
        self.n_mfcc = n_mfcc
        self.max_length = max_length

        # Pre-load MFCC transform once
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=self.n_mfcc
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # ✅ Index safety
        if idx >= len(self.file_list):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.file_list)}")

        audio_path = Path(self.file_list[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"⚠️ Error loading {audio_path}: {e}")
            # Return dummy tensor instead of crashing
            dummy = torch.zeros((self.max_length, self.n_mfcc))
            return dummy, label

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if necessary
        if sr != 16000:
            resample = torchaudio.transforms.Resample(sr, 16000)
            waveform = resample(waveform)

        # Extract MFCC → (n_mfcc, time)
        mfcc = self.mfcc_transform(waveform).squeeze(0)

        # Pad or trim to fixed max_length
        if mfcc.shape[1] < self.max_length:
            pad_len = self.max_length - mfcc.shape[1]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_len))
        else:
            mfcc = mfcc[:, :self.max_length]

        # Transpose to (time, n_mfcc) for model input
        mfcc = mfcc.transpose(0, 1)

        return mfcc, label
