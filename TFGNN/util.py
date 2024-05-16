import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from utils.functions import *





class DataLoader(object):
    def __init__(self, xw, xd, xc,  ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xw) % batch_size)) % batch_size
            x_padding_w = np.repeat(xw[-1:], num_padding, axis=0)  #最后一个元素重复了num_padding次，凑足了batch
            x_padding_d = np.repeat(xd[-1:], num_padding, axis=0)
            x_padding_c = np.repeat(xc[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xw = np.concatenate([xw, x_padding_w], axis=0)
            xd = np.concatenate([xd, x_padding_d], axis=0)
            xc = np.concatenate([xc, x_padding_c], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xw)
        self.num_batch = int(self.size // self.batch_size)
        self.xw = xw
        self.xd = xd
        self.xc = xc
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xw, xd, xc, ys = self.xw[permutation], self.xd[permutation], self.xc[permutation], self.ys[permutation]
        self.xw = xw
        self.xd = xd
        self.xc = xc
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_w = self.xw[start_ind: end_ind, ...]
                x_d = self.xd[start_ind: end_ind, ...]
                x_c = self.xc[start_ind: end_ind, ...]
                y_r = self.ys[start_ind: end_ind, ...]

                yield(x_w, x_d, x_c, y_r)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def accuracy(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    y_flat = y.reshape(y.size(0), -1)
    pred_flat = pred.reshape(pred.size(0), -1)
    return 1 - torch.linalg.norm(y_flat - pred_flat, "fro") / torch.linalg.norm(y_flat, "fro")


def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    y_flat = y.reshape(y.size(0), -1)
    pred_flat = pred.reshape(pred.size(0), -1)
    return 1 - torch.sum((y_flat - pred_flat) ** 2) / torch.sum((y_flat - torch.mean(pred_flat)) ** 2)


def explained_variance(pred, y):
    y_flat = y.reshape(y.size(0), -1)
    pred_flat = pred.reshape(pred.size(0), -1)
    return 1 - torch.var(y_flat - pred_flat) / torch.var(y_flat)


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat

def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj

def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}

    x_all =[]
    y_all =[]

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        x_all.append(cat_data['x'])
        y_all.append(cat_data['y'])
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    x_w = []
    x_w.append(x_all[:-6036, :, :, :]) #-24*12*7*3+12
    x_w.append(x_all[2016:-4020, :, :, :])##-24*12*7*2+12
    x_w.append((x_all[4032:-2004, :, :, :]))#-24*12*7+12
    x_w = np.concatenate(x_w, axis=1)

    x_d=[]
    x_d.append(x_all[4896:-1140, :, :, :])#24*12*7*2+24*12*3:-24*12*4+12
    x_d.append(x_all[5184:-852])#5760+24*12:-24*12*3+12
    x_d.append(x_all[5472:-564])#5760+24*12*2:-24*12*2+12
    x_d = np.concatenate(x_d, axis=1)
    x_c = []
    x_c.append(x_all[6012:-24])#24*12*7*3-12*3
    x_c.append(x_all[6024:-12])
    x_c.append(x_all[6036:])
    x_c = np.concatenate(x_c, axis=1)
    y = y_all[6036:, :, :, :]
    n = y.shape[0]
    n_train = int(np.floor(n*0.7))
    n_val = int(np.floor(n*0.1))
    n_test = n- n_train- n_val
    #for category in ['train', 'val', 'test']:

    data['x_train_w'] = x_w[:n_train,:,:,:]
    data['x_val_w'] = x_w[n_train:n_train+n_val, :, :, :]
    data['x_test_w'] = x_w[n_train + n_val:, :, :, :]

    data['x_train_d'] = x_d[:n_train, :, :, :]
    data['x_val_d'] = x_d[n_train:n_train + n_val, :, :, :]
    data['x_test_d'] = x_d[n_train + n_val:, :, :, :]

    data['x_train_c'] = x_c[:n_train, :, :, :]
    data['x_val_c'] = x_c[n_train:n_train + n_val, :, :, :]
    data['x_test_c'] = x_c[n_train + n_val:, :, :, :]

    data['y_train'] = y[:n_train, :, :, :]
    data['y_val'] = y[n_train:n_train + n_val, :, :, :]
    data['y_test'] = y[n_train + n_val:, :, :, :]
    scaler = StandardScaler(mean=data['x_train_c'][..., 0].mean(), std=data['x_train_c'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        for period in ['w', 'd', 'c']:
            data['x_' + category+'_'+period][..., 0] = scaler.transform(data['x_' + category+'_'+period][..., 0])
    data['train_loader'] = DataLoader(data['x_train_w'], data['x_train_d'], data['x_train_c'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val_w'], data['x_val_d'], data['x_val_c'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test_w'], data['x_test_d'], data['x_test_c'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def load_datasetCSV(dataset_dir, batch_size, train_ratio, val_ratio, seq_len, pre_len, week_len, valid_batch_size= None, test_batch_size=None):
    data = {}
    _x_all = []
    _y_all = []
    _feat = load_features(os.path.join(dataset_dir, 'sz_speed.csv'))
    time_len = _feat.shape[0]

    for i in range(len(_feat) - seq_len - pre_len):
        _x_all.append(np.array(_feat[i : i + seq_len]))
        _y_all.append(np.array(_feat[i + seq_len : i + seq_len + pre_len]))

    x_all = np.array([np.expand_dims(x, axis=-1) for x in _x_all])
    y_all = np.array([np.expand_dims(x, axis=-1) for x in _y_all])

    x_w = []
    x_d = []

    x_c= []
    for i in range(1, week_len):
        x_w.append(x_all[24*seq_len*7*(i-1):-24*seq_len*7*(week_len-i+1)+seq_len, :, :, :])  # -24*4*7*0   -24*4*7*1+4=-668  -24*12*7*3+12
        x_d.append(x_all[24*seq_len*7*(week_len-1)+24*seq_len*(6-week_len+i-1):-24*seq_len*(week_len-i+1+1)+seq_len, :, :, :]) # 24*12*7*2+24*12*3:-24*12*4+12
        x_c.append(x_all[24*seq_len*7*week_len-seq_len*(week_len-i+1):-seq_len*(week_len-i),:,:,:])


    if week_len == 1:
        i = 1
    else:
        i = i+1
    x_w.append(x_all[24 * seq_len * 7 * (i - 1):-24 * seq_len * 7 * (week_len - i + 1) + seq_len, :, :,
               :])  # -24*4*7*0   -24*4*7*1+4=-668  -24*12*7*3+12
    x_d.append(x_all[24 * seq_len * 7 * (week_len - 1) + 24 * seq_len * (6 - week_len + i - 1):-24 * seq_len * (
                week_len - i + 1 + 1) + seq_len, :, :, :])  # 24*12*7*2+24*12*3:-24*12*4+12
    x_c.append(x_all[24 * seq_len * 7 * week_len - seq_len * (week_len - i + 1):, :, :, :])

    x_w = np.concatenate(x_w, axis=1)
    x_d = np.concatenate(x_d, axis=1)
    #_x_c = np.concatenate(_x_c, axis=1)
    #x_c =[_x_c]
    #x_c.append(x_all[24 * seq_len * 7 * week_len - seq_len * (week_len - i):, :, :, :])
    x_c = np.concatenate(x_c, axis=1)

    y = y_all[24*seq_len*7*week_len-seq_len:, :, :, :]
    n = y.shape[0]
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val

    data['x_train_w'] = x_w[:n_train, :, :, :]
    data['x_val_w'] = x_w[n_train:n_train + n_val, :, :, :]
    data['x_test_w'] = x_w[n_train + n_val:, :, :, :]

    data['x_train_d'] = x_d[:n_train, :, :, :]
    data['x_val_d'] = x_d[n_train:n_train + n_val, :, :, :]
    data['x_test_d'] = x_d[n_train + n_val:, :, :, :]

    data['x_train_c'] = x_c[:n_train, :, :, :]
    data['x_val_c'] = x_c[n_train:n_train + n_val, :, :, :]
    data['x_test_c'] = x_c[n_train + n_val:, :, :, :]

    data['y_train'] = y[:n_train, :, :, :]
    data['y_val'] = y[n_train:n_train + n_val, :, :, :]
    data['y_test'] = y[n_train + n_val:, :, :, :]
    scaler = StandardScaler(mean=data['x_train_c'][..., 0].mean(), std=data['x_train_c'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        for period in ['w', 'd', 'c']:
            data['x_' + category + '_' + period][..., 0] = scaler.transform(
                data['x_' + category + '_' + period][..., 0])
    data['train_loader'] = DataLoader(data['x_train_w'], data['x_train_d'], data['x_train_c'], data['y_train'],
                                      batch_size)
    data['val_loader'] = DataLoader(data['x_val_w'], data['x_val_d'], data['x_val_c'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test_w'], data['x_test_d'], data['x_test_c'], data['y_test'],
                                     test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):# 这个地方应该传过来的维度是多少   pred  real
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    acc = accuracy(pred,real)
    r22 = r2(pred, real)
    var = explained_variance(pred, real)
    return mae,mape,rmse, acc.cpu().detach().numpy(), r22.cpu().detach().numpy(), var.cpu().detach().numpy()


def load_dataset_sz(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}

    x_all =[]
    y_all =[]

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        x_all.append(cat_data['x'])
        y_all.append(cat_data['y'])
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    x_w = []
    x_w.append(x_all[:-6036, :, :, :]) #-24*12*7*3+12
    x_w.append(x_all[2016:-4020, :, :, :])##-24*12*7*2+12
    x_w.append((x_all[4032:-2004, :, :, :]))#-24*12*7+12
    x_w = np.concatenate(x_w, axis=1)
    #data['x_w'] = data['x_all'][:-6048, :, :, :]  # -6048
    #data['x_d'] = data['x_all'][5760:-276, :, :, :]#24*12*7*3-24*12*4:-24*12*4=1152
    #data['x_c'] = data['x_all'][6048-36:-36, :, :, :]#12*24*3*7-36:-36
    x_d=[]
    x_d.append(x_all[4896:-1140, :, :, :])#24*12*7*2+24*12*3:-24*12*4+12
    x_d.append(x_all[5184:-852])#5760+24*12:-24*12*3+12
    x_d.append(x_all[5472:-564])#5760+24*12*2:-24*12*2+12
    x_d = np.concatenate(x_d, axis=1)
    x_c = []
    x_c.append(x_all[6012:-24])#24*12*7*3-12*3
    x_c.append(x_all[6024:-12])
    x_c.append(x_all[6036:])
    x_c = np.concatenate(x_c, axis=1)
    y = y_all[6036:, :, :, :]
    n = y.shape[0]
    n_train = int(np.floor(n*0.7))
    n_val = int(np.floor(n*0.1))
    n_test = n- n_train- n_val
    #for category in ['train', 'val', 'test']:

    data['x_train_w'] = x_w[:n_train,:,:,:]
    data['x_val_w'] = x_w[n_train:n_train+n_val, :, :, :]
    data['x_test_w'] = x_w[n_train + n_val:, :, :, :]

    data['x_train_d'] = x_d[:n_train, :, :, :]
    data['x_val_d'] = x_d[n_train:n_train + n_val, :, :, :]
    data['x_test_d'] = x_d[n_train + n_val:, :, :, :]

    data['x_train_c'] = x_c[:n_train, :, :, :]
    data['x_val_c'] = x_c[n_train:n_train + n_val, :, :, :]
    data['x_test_c'] = x_c[n_train + n_val:, :, :, :]

    data['y_train'] = y[:n_train, :, :, :]
    data['y_val'] = y[n_train:n_train + n_val, :, :, :]
    data['y_test'] = y[n_train + n_val:, :, :, :]
    scaler = StandardScaler(mean=data['x_train_c'][..., 0].mean(), std=data['x_train_c'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        for period in ['w', 'd', 'c']:
            data['x_' + category+'_'+period][..., 0] = scaler.transform(data['x_' + category+'_'+period][..., 0])
    data['train_loader'] = DataLoader(data['x_train_w'], data['x_train_d'], data['x_train_c'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val_w'], data['x_val_d'], data['x_val_c'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test_w'], data['x_test_d'], data['x_test_c'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data