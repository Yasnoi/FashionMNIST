import torch
import torch.optim as optim
from torch import nn
import os

from models.net import Net
from utils.data_loader import fashion_mnist_data_loader


class EvaluateNet:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['model']['device'])
        self.data_loader = fashion_mnist_data_loader(config, mode='test')

        self.model = Net().to(self.device)
        checkpoint_dir = config['output']['checkpoint_dir']
        model_save_name = config['output'].get('model_save_name', 'Fashion_MNIST_model.pt')
        model_path = os.path.join(checkpoint_dir, model_save_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Error: Model checkpoint not found at {model_path}')
        self.model.load_state_dict(torch.load(model_path))

        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                test_loss += loss.item() * data.size(0)

                predictions = output.argmax(dim=1, keepdim=True)
                correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()
                total_samples += target.shape[0]

        total_loss = test_loss / total_samples
        total_accuracy = 100. * correct_predictions / total_samples
        print(f'Running on test set...\n'
              f'Test Loss: {total_loss:.4f}, Test Accuracy: {total_accuracy:.2f}%')
