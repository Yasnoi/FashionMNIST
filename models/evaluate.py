import torch
from torch import nn
import os
import csv

from models.net import Net
from utils.data_loader import fashion_mnist_data_loader


class EvaluateNet:
    def __init__(self, config):
        self.config = config
        self.device = self.config['model']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_loader = fashion_mnist_data_loader(config, mode='test')

        self.model = Net().to(self.device)
        checkpoint_dir = config['output']['checkpoint_dir']
        model_save_name = input('Enter model save name: ')
        model_path = os.path.join(checkpoint_dir, model_save_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Error: Model checkpoint not found at {model_path}')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.criterion = nn.CrossEntropyLoss()

        self.results_save_path = os.path.join(config['output']['result_dir'], "submission.csv")

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct_predictions = 0
        total_samples = 0

        result = []
        current_id = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                test_loss += loss.item() * data.size(0)

                predictions = output.argmax(dim=1, keepdim=True)
                correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()
                total_samples += target.shape[0]

                pred_list = predictions.cpu().numpy().flatten()
                for label in pred_list:
                    result.append([current_id, label])
                    current_id += 1

        total_loss = test_loss / total_samples
        total_accuracy = 100. * correct_predictions / total_samples
        print(f'Running on test set...\n'
              f'Test Loss: {total_loss:.4f}, Test Accuracy: {total_accuracy:.2f}%')

        with open(self.results_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'label'])
            writer.writerows(result)
        print(f'Results saved to {self.results_save_path}')
