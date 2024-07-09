import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from parse_arg import get_args
from metrics import AdaCos, ArcFace, SphereFace, CosFace
from automodel import HFModel, HFFeatureExtractor
from dataset import custom_collate_fn, AudioDataset
from utils import attentive_statistics_pooling, EmbeddingCompressor

import warnings
warnings.filterwarnings('ignore')

args = get_args()
device = torch.device("cuda")
best_model_path_waveform = 'cat_emb/result/tw_3/best_model_w.pt'
best_model_path_spectrogram = 'cat_emb/result/tw_3/best_model_s.pt'
best_model_path_compressor = 'cat_emb/result/tw_3/best_model_c.pt'

# Data
train_data = pd.read_csv('./processed_data/evaluation_train.csv')
test_data = pd.read_csv('./processed_data/evaluation_test.csv')
eval_data = pd.read_csv('./processed_data/evaluation_eval.csv')
        
output_size = 94
        
# transformers.AutoProcessor
process_func_w = HFFeatureExtractor('waveform', args)
process_func_s = HFFeatureExtractor('spectrogram', args)

# Custom pytorch Dataset
train_dataset = AudioDataset(file_list=train_data['audio_path'], y=None)
test_dataset = AudioDataset(file_list=test_data['audio_path'], y=None)
eval_dataset = AudioDataset(file_list=eval_data['audio_path'], y=None)

# transformers.AutoModel
model_waveform = HFModel('waveform', args, None, output_size).to(device)
model_spectrogram = HFModel('spectrogram', args, None, output_size).to(device)
att_pool = attentive_statistics_pooling(768, 768).to(device)
emb_comp = EmbeddingCompressor().to(device)

model_waveform = nn.DataParallel(model_waveform)
model_spectrogram = nn.DataParallel(model_spectrogram)
att_pool = nn.DataParallel(att_pool)
emb_comp = nn.DataParallel(emb_comp)

# Dataloader
train_loader = DataLoader(train_dataset, 
                          batch_size= args.batch_size, 
                          shuffle=False, 
                          num_workers=args.num_workers, 
                          collate_fn=custom_collate_fn)

test_loader = DataLoader(test_dataset, 
                         batch_size=args.batch_size, 
                         shuffle=False, 
                         num_workers=args.num_workers, 
                         collate_fn=custom_collate_fn)

eval_loader = DataLoader(eval_dataset, 
                         batch_size=args.batch_size, 
                         shuffle=False, 
                         num_workers=args.num_workers, 
                         collate_fn=custom_collate_fn)

model_waveform.load_state_dict(torch.load(best_model_path_waveform))
model_spectrogram.load_state_dict(torch.load(best_model_path_spectrogram))
emb_comp.load_state_dict(torch.load(best_model_path_compressor))

model_waveform.eval()
model_spectrogram.eval()
emb_comp.eval()

with torch.no_grad():
    result_train = []
    for batch in tqdm(train_loader):
        x, y = batch
        x_w = process_func_w(x)
        x_s = process_func_s(x)
        x_w, x_s, y = x_w.to(device), x_s.to(device), y.to(device)
        output_waveform = model_waveform(x_w)
        output_spectrogram = model_spectrogram(x_s)

        wav_mean, wav_std = att_pool(output_waveform)
        ast_cls = output_spectrogram[:, 0, :]
        
        feature = torch.cat((wav_mean.unsqueeze(1), wav_std.unsqueeze(1), ast_cls.unsqueeze(1)), dim=1)
        feature_compressed = emb_comp(feature)
             
        result_train.append(feature_compressed.cpu().numpy())
        
result_train = np.array(result_train).reshape(-1, 512)
with open('cat_emb/result_emb/train_emb.pickle', 'wb') as f:
    pickle.dump(result_train, f)
f.close()

with torch.no_grad():
    result_test = []
    for batch in tqdm(test_loader):
        x, y = batch
        x_w = process_func_w(x)
        x_s = process_func_s(x)
        x_w, x_s, y = x_w.to(device), x_s.to(device), y.to(device)
        output_waveform = model_waveform(x_w)
        output_spectrogram = model_spectrogram(x_s)

        wav_mean, wav_std = att_pool(output_waveform)
        ast_cls = output_spectrogram[:, 0, :]
        
        feature = torch.cat((wav_mean.unsqueeze(1), wav_std.unsqueeze(1), ast_cls.unsqueeze(1)), dim=1)
        feature_compressed = emb_comp(feature)
             
        result_test.extend(feature_compressed.cpu().tolist())
        
result_test = np.array(result_test).reshape(-1, 512)
with open('cat_emb/result_emb/test_emb.pickle', 'wb') as f:
    pickle.dump(result_test, f)
f.close()

with torch.no_grad():
    result_eval = []
    for batch in tqdm(eval_loader):
        x, y = batch
        x_w = process_func_w(x)
        x_s = process_func_s(x)
        x_w, x_s, y = x_w.to(device), x_s.to(device), y.to(device)
        output_waveform = model_waveform(x_w)
        output_spectrogram = model_spectrogram(x_s)

        wav_mean, wav_std = att_pool(output_waveform)
        ast_cls = output_spectrogram[:, 0, :]
        
        feature = torch.cat((wav_mean.unsqueeze(1), wav_std.unsqueeze(1), ast_cls.unsqueeze(1)), dim=1)
        feature_compressed = emb_comp(feature)
             
        result_eval.extend(feature_compressed.cpu().tolist())

result_eval = np.array(result_eval).reshape(-1, 512)
with open('cat_emb/result_emb/eval_emb.pickle', 'wb') as f:
    pickle.dump(result_eval, f)
f.close()
