import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoProcessor, AutoModel, AutoFeatureExtractor

class HFFeatureExtractor(nn.Module):
    def __init__(self, flag, args):
        super().__init__()
        # choose pretrained model according to flag
        if flag=='spectrogram':
            pretrained_model = args.pretrained_model_spectrogram
        elif flag=='waveform':
            pretrained_model = args.pretrained_model_waveform
        else:
            raise TypeError
        
        # initiate AutoProcessor
        self.extractor = AutoProcessor.from_pretrained(pretrained_model, sampling_rate=args.sr)
        self.sr = args.sr
        
    def forward(self, x):
        # Need attention mask as well in order to do attention pooling
        return self.extractor(x, sampling_rate=self.sr, padding=True, return_tensors='pt')['input_values']
    
class HFModel(nn.Module):
    def __init__(self, flag, args, input_size, output_size):
        super().__init__()
        # choose pretrained model according to flag
        if flag=='spectrogram':
            pretrained_model = args.pretrained_model_spectrogram
        elif flag=='waveform':
            pretrained_model = args.pretrained_model_waveform
        else:
            raise TypeError
        # initialte AutoModel
        self.model = AutoModel.from_pretrained(pretrained_model, num_labels=output_size, ignore_mismatched_sizes=True)

    def forward(self, x):
        # last hidden state to concat embeddings from two different models
        return self.model(x).last_hidden_state
    
