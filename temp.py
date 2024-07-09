def train(self):
    best_loss = np.inf
    accumulation_steps = 8
    for epoch in range(1, self.epochs+1):
        self.optimizer_waveform.zero_grad()
        self.optimizer_spectrogram.zero_grad()
        
        total_loss = 0
        for i, batch in tqdm(enumerate(self.train_loader), file=sys.stdout):
            x, y = batch
            x_w = self.process_function_w(x)
            x_s = self.process_function_s(x)
            x_w, x_s, y = x_w.to(self.device), x_s.to(self.device), y.to(self.device)
            
            output_waveform = self.model_waveform(x_w)
            output_spectrogram = self.model_spectrogram(x_s)
            
            output_waveform_pooled = torch.mean(output_waveform, dim=1) # mean pooling
            output_spectrogram_pooled = output_spectrogram[:, 0, :]
            # need attention pooling
            # output = self.classifier(torch.cat((output_waveform_pooled, output_spectrogram_pooled), dim=1))
            feature = torch.cat((output_waveform_pooled, output_spectrogram_pooled), dim=1)
            output = self.metric_fn(feature, y)
            loss = self.loss_fn(output, y)
            loss = loss / accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * x_w.shape[0]
            
            if (i+1) % accumulation_steps == 0:
                self.optimizer_waveform.step()
                self.optimizer_spectrogram.step()

                self.optimizer_waveform.zero_grad()
                self.optimizer_spectrogram.zero_grad()
            
        if (i+1) % accumulation_steps != 0:
            self.optimizer_waveform.step()
            self.optimizer_spectrogram.step()

            self.optimizer_waveform.zero_grad()
            self.optimizer_spectrogram.zero_grad()
            
        self.scheduler_w.step()
        self.scheduler_s.step()
        
        loss_train = total_loss / self.len_train
        loss_val = self.valid_step()
        
        self.logger.info(f"Epoch {str(epoch).zfill(5)}: t_loss: {loss_train:.3f} v_loss: {loss_val:3f}")
        
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(self.model_waveform.state_dict(), self.best_model_path_w)
            torch.save(self.model_spectrogram.state_dict(), self.best_model_path_s)
            # torch.save(self.classifier.state_dict(), self.best_model_path_c)
            patient_count = 0
        else:
            patient_count += 1
            
        if patient_count == self.patience:
            break