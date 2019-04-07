import torch
import numpy as np
import pickle
import covariance_matrix as cm
from model_classes import *


def load_object(file_name):
    """load the pickled object"""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def view_data(data_path):
    data = load_object(data_path)
    prices = data['prices']
    names = data['features']['names']
    features = data['features']['values']
    # print(prices.shape)
    # print(names)
    # print(features.shape)
    return prices, features


class Strategy():
    def __init__(self):
        self.all_price_data, self.all_feature_data = view_data('C3_train.pkl')
        self.all_price_data = torch.from_numpy(self.all_price_data).clone().float()[:-1,:]
        self.all_feature_data = torch.from_numpy(self.all_feature_data).clone().float()
        self.all_return_data = torch.zeros(self.all_price_data.size(0)-1, self.all_price_data.size(1))
        for col in range(self.all_price_data.size(1)):
            self.all_return_data[:,col] = cm.calc_log_returns(self.all_price_data[:,col])

        self.models = []
        for ticker in range(680):
            stock_model = torch.nn.Sequential(
                LinearLayer(2,2),
                ProductLayer(2,1),
                LogOutput(),
            )
            stock_model.load_state_dict(torch.load(f'final_models/{ticker}.pt'))
            self.models.append(stock_model)

    # add new numpy data to tensor data
    def update_data(self, price_data, feature_data):
        self.all_price_data = torch.cat( (self.all_price_data, torch.from_numpy(price_data).float().unsqueeze(0) ) , dim=0)
        self.all_feature_data = torch.cat( (self.all_feature_data, torch.from_numpy(feature_data).float().unsqueeze(0) ) , dim=0)
        new_return_data = torch.zeros(self.all_price_data.size(0)-1, self.all_price_data.size(1))
        for col in range(self.all_price_data.size(1)):
            new_return_data[:,col] = cm.calc_log_returns(self.all_price_data[:,col])
        self.all_return_data = new_return_data.clone()

    def handle_update(self, inx, price, factors):
        """Put your logic here
        Args:
            inx: zero-based inx in days
            price: [num_assets, ]
            factors: [num_assets, num_factors]
        Return:
            allocation: [num_assets, ]
        """
        self.update_data(price, factors)
        all_rsi_data = self.all_feature_data[:,:,4]

        predicted_returns = []
        for ticker in range(680):
            rsi_data = all_rsi_data[:,ticker]*.01
            # current, previous
            rsi_inputs = torch.tensor([rsi_data[-1], rsi_data[-2]])
            model = self.models[ticker]
            pred_return = model(rsi_inputs).item()
            predicted_returns.append(pred_return)

        covariances = cm.cov_matrix(self.all_return_data)

        pred_returns_array = np.array(predicted_returns)#.reshape(1,-1)
        allocation, _, _ = cm.optimal_portfolio(pred_returns_array, 4, covariances)
        allocation = allocation.astype(np.float).flatten()
        assert price.shape[0] == factors.shape[0]
        # return np.array([1.0] * price.shape[0])
        return allocation
