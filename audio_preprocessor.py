import torch
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def preprocess_audio(waveform, sr, target_sr=16000, target_sec=3.0):
    """Preprocess an input waveform for deep learning models."""
    print(f"Input waveform shape: {waveform.shape}, sr: {sr}")
    
    num_samples = int(target_sr * target_sec)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print(f"Converted to mono: {waveform.shape}")
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        print(f"Resampled to {target_sr}Hz: {waveform.shape}")
    
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
        print(f"Normalized (max was {max_val:.4f}): {waveform.shape}")
    else:
        print(f"Warning: Audio is silent (max value is 0)")

    if waveform.shape[1] < num_samples:
        pad = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
        print(f"Padded by {pad} samples: {waveform.shape}")
    else:
        waveform = waveform[:, :num_samples]
        print(f"Trimmed to {num_samples} samples: {waveform.shape}")

    print(f"Final preprocessed shape: {waveform.shape}")
    return waveform


def preprocess_from_csv(csv_path, input_dir, output_root, label_map=None, use_full_path=False, class_limit=None):
    """Preprocess audio files listed in a CSV file and save them as torch tensors"""
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    output_real = output_root / "real_processed"
    output_fake = output_root / "fake_processed"
    output_real.mkdir(parents=True, exist_ok=True)
    output_fake.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    real = 0
    fake = 0

    print(f"Preprocessing from CSV: {csv_path}")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if class_limit and real >= class_limit and fake >= class_limit:
            break
        file_key = row[0]
        label_raw = row[1]

        if label_map:
            label = label_map[label_raw.lower()]
        else:
            label = 0 if label_raw.lower() == "bona-fide" or label_raw.lower() == "real" else 1

        if label == 0 and real >= class_limit:
            continue
        if label == 1 and fake >= class_limit:
            continue

        if use_full_path:
            audio_path = Path(file_key)
        else:
            if file_key.endswith(".flac") or file_key.endswith(".wav"):
                audio_path = input_dir / file_key
            else:
                audio_path = input_dir / (file_key + ".flac")

        save_dir = output_real if label == 0 else output_fake
        save_path = save_dir / (audio_path.stem + ".pt")

        if save_path.exists():
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            waveform = preprocess_audio(waveform, sr)
            torch.save(waveform, save_path)
            if label == 0:
                real += 1
            else:
                fake += 1
        except Exception as e:
            print(f"Error with {audio_path.name}: {e}")

    print(f"Completed: {len(list(output_real.glob('*.pt')))} real, {len(list(output_fake.glob('*.pt')))} fake saved.")