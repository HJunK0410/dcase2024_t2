import os 
import sys
import torch
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self, train_loader, valid_loader, 
                 process_function_w, process_function_s, 
                 model_waveform, model_spectrogram, 
                 metric_fn, loss_fn, 
                 optimizer_waveform, optimizer_spectrogram, optimizer_compressor, 
                 scheduler_w, scheduler_s, scheduler_c, 
                 att_pool, emb_comp, 
                 device, patience, epochs, result_path, logger, len_train, len_valid):
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.process_function_w = process_function_w
        self.process_function_s = process_function_s
        self.model_waveform = model_waveform
        self.model_spectrogram = model_spectrogram
        self.metric_fn = metric_fn
        self.loss_fn = loss_fn
        self.optimizer_waveform = optimizer_waveform
        self.optimizer_spectrogram = optimizer_spectrogram
        self.optimizer_compressor = optimizer_compressor
        self.scheduler_w = scheduler_w
        self.scheduler_s = scheduler_s
        self.scheduler_c = scheduler_c
        self.att_pool = att_pool
        self.emb_comp = emb_comp
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = logger
        self.best_model_path_w = os.path.join(result_path, 'best_model_w.pt')
        self.best_model_path_s = os.path.join(result_path, 'best_model_s.pt')
        self.best_model_path_c = os.path.join(result_path, 'best_model_c.pt')
        self.len_train = len_train
        self.len_valid = len_valid

    def train(self):
        best_loss = np.inf
        accumulation_steps = 4
        self.model_waveform.train()
        self.model_spectrogram.train()
        self.emb_comp.train()
        
        for epoch in range(1, self.epochs+1):
            self.optimizer_waveform.zero_grad()
            self.optimizer_spectrogram.zero_grad()
            self.optimizer_compressor.zero_grad()
            
            total_loss = 0
            for i, batch in enumerate(tqdm(self.train_loader, file=sys.stdout)):
                x, y = batch
                x_w = self.process_function_w(x)
                x_s = self.process_function_s(x)
                x_w, x_s, y = x_w.to(self.device), x_s.to(self.device), y.to(self.device)
                
                output_waveform = self.model_waveform(x_w)
                output_spectrogram = self.model_spectrogram(x_s)

                wav_mean, wav_std = self.att_pool(output_waveform)
                ast_cls = output_spectrogram[:, 0, :]
                
                feature = torch.cat((wav_mean.unsqueeze(1), wav_std.unsqueeze(1), ast_cls.unsqueeze(1)), dim=1)
                feature_compressed = self.emb_comp(feature)
                                
                output = self.metric_fn(feature_compressed, y)                
                loss = self.loss_fn(output, y)
                loss = loss / accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * x_w.shape[0]
                
                if (i+1) % accumulation_steps == 0:
                    self.optimizer_waveform.step()
                    self.optimizer_spectrogram.step()
                    self.optimizer_compressor.step()
                    
                    self.optimizer_waveform.zero_grad()
                    self.optimizer_spectrogram.zero_grad()
                    self.optimizer_compressor.zero_grad()
                    
                # if i%100==1:    
                #     print(f'Current Train Loss: {total_loss/(x_w.shape[0]*i):.3f}')
                
            if (i+1) % accumulation_steps != 0:
                self.optimizer_waveform.step()
                self.optimizer_spectrogram.step()
                self.optimizer_compressor.step()

                self.optimizer_waveform.zero_grad()
                self.optimizer_spectrogram.zero_grad()
                self.optimizer_compressor.zero_grad()

            loss_train = total_loss / self.len_train
            loss_val = self.valid_step()
            
            # self.scheduler_w.step(loss_val)
            # self.scheduler_s.step(loss_val)
            # self.scheduler_c.step(loss_val)
            
            self.scheduler_w.step()
            self.scheduler_s.step()
            self.scheduler_c.step()
            
            self.logger.info(f"Epoch {str(epoch).zfill(5)}: t_loss: {loss_train:.5f} v_loss: {loss_val:.5f}")
            
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(self.model_waveform.state_dict(), self.best_model_path_w)
                torch.save(self.model_spectrogram.state_dict(), self.best_model_path_s)
                torch.save(self.emb_comp.state_dict(), self.best_model_path_c)
                patient_count = 0
            else:
                patient_count += 1
                
            if patient_count == self.patience:
                break
    
    def valid_step(self):
        self.model_waveform.eval()
        self.model_spectrogram.eval()
        self.emb_comp.eval()
        
        with torch.no_grad():
            total_loss = 0
            for batch in self.valid_loader:
                x, y = batch
                x_w = self.process_function_w(x)
                x_s = self.process_function_s(x)
                x_w, x_s, y = x_w.to(self.device), x_s.to(self.device), y.to(self.device)
                output_waveform = self.model_waveform(x_w)
                output_spectrogram = self.model_spectrogram(x_s)
                
                wav_mean, wav_std = self.att_pool(output_waveform)
                ast_cls = output_spectrogram[:, 0, :]
                
                feature = torch.cat((wav_mean.unsqueeze(1), wav_std.unsqueeze(1), ast_cls.unsqueeze(1)), dim=1)
                feature_compressed = self.emb_comp(feature)
                                
                output = self.metric_fn(feature_compressed, y)       
                loss = self.loss_fn(output, y)
                total_loss += loss.item() * x_w.shape[0]
                
        return total_loss/self.len_valid