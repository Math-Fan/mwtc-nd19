import torch
# from strategy import view_data
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers

# import matplotlib.pyplot as plt

#for testing
# import datetime

# all_price_data_np, all_feature_data_np, feature_names_np = view_data('C3_train.pkl')
# all_price_data = torch.from_numpy(all_price_data_np).clone().float()
# all_feature_data = torch.from_numpy(all_feature_data_np).clone().float()
day = 756

def update_data(price_arr, feature_arr):
    new_price_data = torch.cat(np.array([all_price_data, torch.from_numpy(price_arr).clone().float()]))
    new_feature_data = torch.cat(np.array([all_feature_data, torch.from_numpy(feature_arr).clone().float()]))


# takes a price or feature vector and gives log returns vector of size 1 smaller
def calc_log_returns(vector):
    return (vector[1:] / vector[:-1]).log()

def covariance(x,y):
    return ((x-x.mean())*(y-y.mean())).sum() / (x.size(0)-1)

def cov_matrix(tensor):
    new_tensor = torch.zeros_like(tensor)
    for col in range(tensor.size(1)):
        new_tensor[:,col] = tensor[:,col] - tensor[:,col].mean()
    return torch.mm(new_tensor.t(), new_tensor) / (new_tensor.size(0)-1)


def generate_return_data():
    return_data = torch.zeros(all_price_data.size(0)-1, all_price_data.size(1))
    for ticker in range(all_price_data.size(1)):
        return_data[:,ticker] = calc_log_returns(all_price_data[:,ticker])
    return return_data


def get_return_vec(return_data_t):
    return [sum(return_data_t[i])/len(return_data_t[i]) for i in range(len(return_data_t))]

#
# def get_cov_matrix(return_covariances):
#     return return_covariances.numpy()

# Now solve the Markowitz Problem with our return covariances
def optimal_portfolio(returns, N, covs):
    n = len(returns)
    returns = np.asmatrix(returns).astype(np.double)

    # N = 100
    mus = [10.**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    cov = covs.numpy().astype(np.double)
    S = opt.matrix(cov)
    pbar = opt.matrix(returns.reshape(-1,1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    solvers.options['show_progress'] = False
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    # print(m1[2])
    # print(m1[0])
    # x1 = np.sqrt(m1[2] / m1[0])
    x1=0

    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

# time1 = datetime.datetime.now()
# weights1, returns1, risks1 = optimal_portfolio(return_data_t, 100)
# diff1 = (datetime.datetime.now() - time1).total_seconds()*1000
#
# time2 = datetime.datetime.now()
# weights2, returns2, risks2 = optimal_portfolio(return_data_t, 30)
# diff2 = (datetime.datetime.now() - time2).total_seconds()*1000
#
# plt.plot(stds, means, 'o')
# plt.ylabel('mean')
# plt.xlabel('std')
# plt.plot(risks, returns, 'y-o')
# print(diff1)
# print(diff2)
# plt.show()
#
#
# def portfolio_return(weights, returns):
#     return torch.dot(weights, returns)
#
#
# # REACT TO UPDATE
# def handle(inx, price, factors):
#     day = 756+inx
#     all_price_data, all_feature_data = update_data(price, factors)
#     price_covariances = cov_matrix(all_price_data)
#     return_data = generate_return_data()
#     return_data_t = return_data.t()
#     return_vec = get_return_vec(return_data_t)
#     return_covariances = cov_matrix(return_data)
#     weights, returns, risks = optimal_portfolio(return_data_t, 30, return_covariances)
#     return weights
