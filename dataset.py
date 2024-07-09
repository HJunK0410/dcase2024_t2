import torch
import librosa
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import warnings 
warnings.filterwarnings('ignore')

def load_audio(audio_path, sr=16000):
    waveform, _ = librosa.load(audio_path, sr=sr)
    return waveform

def custom_collate_fn(batch):
    # Extract data and targets from both batches
    audio = [i[0] for i in batch]
    label = [i[1] for i in batch]
    
    return audio, torch.tensor(label)

class AudioDataset(Dataset):
    def __init__(self, file_list, y=None, augment=False):
        self.file_list = file_list
        self.augment = augment
        if y is not None:
            self.y = y
        else:
            self.y = torch.zeros(len(self.file_list)) # for test set
            
    def speed_perturb(self, waveform, speed_range=(0.9, 1.1)):
        speed = random.uniform(*speed_range)
        waveform_stretched = librosa.effects.time_stretch(y=waveform, rate=speed)
        waveform_stretched_fixed = librosa.util.fix_length(data=waveform_stretched, size=len(waveform))
        
        return waveform_stretched_fixed
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):        
        waveform = load_audio(self.file_list[index])
        
        if self.augment:
            waveform_stretched = self.speed_perturb(waveform)
            return waveform_stretched, self.y[index]
        else:
            return waveform, self.y[index]
    
