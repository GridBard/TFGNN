import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, fusion_flag):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, fusion_flag=fusion_flag)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        for i in range(len(input)):
            input[i] = nn.functional.pad(input[i],(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1, 3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        acc = util.accuracy(predict, real)
        r2 = util.r2(predict, real)
        var = util.explained_variance(predict, real)
        return loss.item(), mape, rmse, acc.cpu().detach().numpy(), r2.cpu().detach().numpy(), var.cpu().detach().numpy()

    def eval(self, input, real_val):
        self.model.eval()
        for i in range(len(input)):
            input[i] = nn.functional.pad(input[i], (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        acc = util.accuracy(predict, real)
        r2 = util.r2(predict, real)
        var =util.explained_variance(predict, real)
        return loss.item(), mape, rmse, acc.cpu().detach().numpy(), r2.cpu().detach().numpy(), var.cpu().detach().numpy()

