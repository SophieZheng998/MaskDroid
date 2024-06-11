import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

from collections import defaultdict
from logging import Logger
from torch.utils.data import DataLoader
from DeepRefiner.Network.dataset import ClassifyDataset
from gensim.models import KeyedVectors

from Utils.Network.helper import get_device, eval_metrics, adjust_learning_rate


class LSTM(nn.Module):
    def __init__(self, wordvec_embedding_path: str, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout_ratio: float = 0.2):
        super(LSTM, self).__init__()

        self.embedding = self.load_pretrained_embedding(wordvec_embedding_path)
        self.embedding.weight.requires_grad = True
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.pred_linear = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        x = self.embedding(x)
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (batch_size, hidden_size)
        # Max pooling
        out = torch.max(out, 1).values
        out = self.pred_linear(out)

        loss = self.loss_fn(out, y)
        return loss
    
    def predict(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        out = torch.max(out, 1).values
        out = self.pred_linear(out)

        # print(out)
        y_pred = torch.argmax(out, dim=1)
        return y_pred
    
    def load_pretrained_embedding(self, wordvec_embedding_path: str):
        if not os.path.exists(wordvec_embedding_path):
            raise Exception(f'Word embedding file {wordvec_embedding_path} does not exist.')
        word_vocab = KeyedVectors.load(wordvec_embedding_path, mmap='r')
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_vocab.vectors))

        return embedding
    
def lstm_train(model_path: str, train_data: ClassifyDataset, val_data: ClassifyDataset, test_data: ClassifyDataset, model: LSTM, logger: Logger, **kwargs):
    """
    Train a LSTM model.
    :param model_path: Path to save the model.
    :param train_data: Training dataset.
    :param val_data: Validation dataset.
    :param test_data: Test dataset.
    :param model: LSTM model.
    :param logger: Logger.
    :param kwargs: Other parameters.
    """
    device_id = kwargs['device']
    device = get_device(device_id)
    model = model.to(device)

    epochs = kwargs['epochs']
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']
    batch_size = kwargs['batch_size']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    ret_val = defaultdict(lambda: 0)
    ret_test = defaultdict(lambda: 0)
    ret_val['f1'] = 0.

    # Early stopping
    patience = 9
    tolerance = 0.001
    no_improvement_count = 0
    # Total time
    total_train_time = 0.
    total_val_time = 0.

    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        total_loss = 0.

        time_start = time.time()
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            loss = model(data, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        mean_loss = total_loss / len(train_loader)
        time_end = time.time()
        train_time = time_end - time_start
        total_train_time += train_time

        logger.info(f'Epoch {epoch + 1}/{epochs} | Loss: {mean_loss:.4f} | Time: {train_time:.2f}s')

        time_val_start = time.time()
        ret_val_tmp = lstm_evaluate(val_data, model, **kwargs)
        time_val_end = time.time()
        val_time = time_val_end - time_val_start
        total_val_time += val_time
        logger.info(f'Validation F1: {ret_val_tmp["f1"]:.4f}, Recall: {ret_val_tmp["recall"]:.4f}, Precision: {ret_val_tmp["precision"]:.4f}, Accuracy: {ret_val_tmp["accuracy"]:.4f}, AUC: {ret_val_tmp["auc"]:.4f}, Time: {val_time:.2f}s')
        logger.warning(f'Total training time: {total_train_time:.2f}s, Total validation time: {total_val_time:.2f}s')
        if ret_val_tmp['f1'] > ret_val['f1'] + tolerance:
            ret_val = ret_val_tmp
            test_start_time = time.time()
            ret_test = lstm_evaluate(test_data, model, **kwargs)
            test_end_time = time.time()
            torch.save(model, model_path)
            # Reset early stopping counter
            no_improvement_count = 0

            logger.info(f'Test F1: {ret_test["f1"]:.4f}, Recall: {ret_test["recall"]:.4f}, Precision: {ret_test["precision"]:.4f}, Accuracy: {ret_test["accuracy"]:.4f}, AUC: {ret_test["auc"]:.4f}, Time: {test_end_time - test_start_time:.2f}s')
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                logger.warning(f'Early stopping at epoch {epoch + 1}')
                break
    
    return ret_val, ret_test

def lstm_evaluate(data: ClassifyDataset, model: LSTM, **kwargs):
    """
    Evaluate a LSTM model.
    :param data: Dataset.
    :param model: LSTM model.
    :param kwargs: Other parameters.
    """
    device_id = kwargs['device']
    device = get_device(device_id)
    model = model.to(device)

    batch_size = kwargs['batch_size']
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            pred = model.predict(data)

            y_pred.extend(pred.cpu().numpy())
            y_true.extend(label.cpu().numpy())
            
    return eval_metrics(y_true, y_pred)