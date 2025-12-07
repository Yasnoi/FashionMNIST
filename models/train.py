import torch
import torch.optim as optim
from torch import nn
import os
import copy
from datetime import datetime

from models.net import Net
from utils.data_loader import fashion_mnist_data_loader


class TrainNet:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['model']['device'])
        self.train_data_loader, self.validate_data_loader = fashion_mnist_data_loader(self.config, mode='train')

        self.epochs = config['model']['epochs']
        self.learning_rate = config['model']['learning_rate']
        self.weight_decay = config['model']['weight_decay']

        self.file_name = f"{config['output']['model_save_name']}{datetime.now().strftime('_%Y-%m-%d-%H-%M')}.pt"
        self.save_path = os.path.join(config['output']['checkpoint_dir'], self.file_name)

        self.model = Net().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # variable learning_rate
        schedule_config = config['model']['scheduler']
        if schedule_config['name'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=schedule_config['mode'],
                factor=schedule_config['factor'],
                patience=schedule_config['patience'],
                min_lr=schedule_config['min_lr'],
            )
        else:
            self.scheduler = None

        # early stopping
        self.early_stopping_patience = config['model']['early_stopping']['patience']
        self.early_stopping_min_delta = config['model']['early_stopping']['min_delta']
        self.best_validate_loss = float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_state = None

    def train_epoch(self, data_loader):
        self.model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * data.size(0)

            predictions = output.argmax(dim=1, keepdim=True)
            correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()
            total_samples += target.shape[0]

            if batch_idx % 100 == 0:
                batch_accuracy = 100. * correct_predictions / total_samples
                print(f'Batch: {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}, Current Accuracy: {batch_accuracy:.2f}%')

        epoch_loss = epoch_loss / total_samples
        epoch_accuracy = 100. * correct_predictions / total_samples

        return epoch_loss, epoch_accuracy

    def validate_epoch(self, data_loader):
        self.model.eval()
        validate_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                validate_loss += loss.item() * data.size(0)

                predictions = output.argmax(dim=1, keepdim=True)
                correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()
                total_samples += target.shape[0]

        total_loss = validate_loss / total_samples
        total_accuracy = 100. * correct_predictions / total_samples

        return total_loss, total_accuracy

    def train(self):
        for epoch in range(self.epochs):
            print(f'------------------Start training epoch {epoch + 1}------------------')
            train_loss, train_accuracy = self.train_epoch(self.train_data_loader)
            print(f'------------------Start validating epoch {epoch + 1}------------------')
            validate_loss, validate_accuracy = self.validate_epoch(self.validate_data_loader)
            print(f'Epoch: {epoch + 1}/{self.epochs}\n'
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n'
                  f'Validate Loss: {validate_loss:.4f}, Validate Accuracy: {validate_accuracy:.2f}%')
            # variable learning_rate
            if self.scheduler:
                self.scheduler.step(validate_accuracy)
            # early stopping
            if validate_loss < self.best_validate_loss - self.early_stopping_min_delta:
                self.best_validate_loss = validate_loss
                self.early_stopping_counter = 0
                self.early_stopping_state = copy.deepcopy(self.model.state_dict())
                print(f'Validation loss improved, saving best model.\n'
                      f'Best Loss: {validate_loss:.4f}')
            else:
                self.early_stopping_counter += 1
                print(f'Validation loss did not improve.\n'
                      f'Patience Counter: {self.early_stopping_counter}/{self.early_stopping_patience}')
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f'Early stopping triggered after {self.early_stopping_counter} epochs without improvement.')
                    break

        if self.early_stopping_state:
            self.model.load_state_dict(self.early_stopping_state)
            print(f'Loaded best model state for final saving.')
        else:
            print(f'No improvement recorded, saving the last model state.')

        torch.save(self.model.state_dict(), self.save_path)
        print(f'Model saved to {self.save_path}')
