import os 
import pickle
import tqdm
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from trainer import Trainer
from parse_arg import get_args
from utils import seed_everything, attentive_statistics_pooling, EmbeddingCompressor
from metrics import AdaCos, ArcFace, SphereFace, CosFace
from automodel import HFModel, HFFeatureExtractor
from dataset import custom_collate_fn, AudioDataset

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

#################################################################################################################
#################################################################################################################

if __name__ == "__main__":
    # Environment
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    args.device = device
    args.train_flag = 'machine'

    # Data
    train_data = pd.read_csv(args.train_machine)
    train_data = train_data[train_data['domain']=='source'].reset_index(drop=True)
  
    # Label encoding
    le = LabelEncoder()
    le.fit(train_data['machine_label'])
    train_data['machine_label'] = le.transform(train_data['machine_label'])
    output_size = len(le.classes_)
    with open('le_machine.pickle', 'wb') as f:
        pickle.dump(le, f)
    f.close()

    # Result path
    result_path = os.path.join(args.result_path, f'tm_1')
    os.makedirs(result_path, exist_ok=True)
    
    # Logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))
    logger.info(args)

    # transformers.AutoProcessor
    process_func_w = HFFeatureExtractor('waveform', args)
    process_func_s = HFFeatureExtractor('spectrogram', args)
    
    X_train, X_test, y_train, y_test = train_test_split(
    train_data['audio_path'], train_data['machine_label'], test_size=0.3, random_state=42, stratify=train_data['machine_label']
    )

    # Custom pytorch Dataset
    train_dataset = AudioDataset(file_list=X_train.reset_index(drop=True), y=y_train.values)
    valid_dataset = AudioDataset(file_list=X_test.reset_index(drop=True), y=y_test.values)
    
    # Dataloader
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers, 
                            collate_fn=custom_collate_fn)
    
    valid_loader = DataLoader(valid_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers, 
                            collate_fn=custom_collate_fn)

    # transformers.AutoModel
    model_waveform = HFModel('waveform', args, None, output_size).to(device)
    model_spectrogram = HFModel('spectrogram', args, None, output_size).to(device)
    att_pool = attentive_statistics_pooling(768, 768).to(device)
    emb_comp = EmbeddingCompressor().to(device)

    # Training config
    loss_fn = nn.CrossEntropyLoss() # need to change loss fn 
    
    metric_fn = ArcFace(512, output_size).to(device)
    
    optimizer_waveform = optim.AdamW(model_waveform.parameters(), lr=args.lr)
    optimizer_spectrogram = optim.AdamW(model_spectrogram.parameters(), lr=args.lr)
    optimizer_compressor = optim.AdamW(emb_comp.parameters(), lr=args.lr)

    scheduler_w =  optim.lr_scheduler.LambdaLR(optimizer=optimizer_waveform, lr_lambda=lambda epoch: args.lr_decay ** epoch)
    scheduler_s =  optim.lr_scheduler.LambdaLR(optimizer=optimizer_spectrogram, lr_lambda=lambda epoch: args.lr_decay ** epoch)
    scheduler_c =  optim.lr_scheduler.LambdaLR(optimizer=optimizer_compressor, lr_lambda=lambda epoch: args.lr_decay ** epoch)
    

    
    trainer = Trainer(train_loader, valid_loader, 
                    process_func_w, process_func_s, 
                    model_waveform, model_spectrogram, 
                    metric_fn, loss_fn, 
                    optimizer_waveform, optimizer_spectrogram, optimizer_compressor, 
                    scheduler_w, scheduler_s, scheduler_c, 
                    att_pool, emb_comp, 
                    device, args.patience, args.epochs, result_path, logger, len(train_dataset), len(valid_dataset))
    
    print(">>>>>>>>>> Start Training >>>>>>>>>>")
    trainer.train()