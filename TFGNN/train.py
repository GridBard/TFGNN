import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import os

parser = argparse.ArgumentParser()
#parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='data/shenzhen',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=4,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=156,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/shenzhen',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--fusion',type=str,default='hour') #start, skip, week, day, hour
parser.add_argument("--gpuid", type=str, default='0')
parser.add_argument("--predtime", type=int, default=60)

parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.1)
parser.add_argument('--seq_len', type=int, default=4, help='total train seq for a hour')
parser.add_argument('--pre_len', type=int, default=4, help='pre seq for a hour')
parser.add_argument('--week_len', type=int, default=1, help='pre seq for a hour')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
#args.device = torch.device("cuda:{}".format(args.gpuid)) #自己在服务器运行，需要分配gpuid
#args.device = torch.device("cuda") #bsub <  提交作业，不需要自己分配gpuid
args.device = torch.device("cpu")
print('device:{}'.format(args.device))

def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)

    if args.data == 'data/shenzhen':
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
        dataloader = util.load_datasetCSV(args.data, args.batch_size, args.train_ratio, args.val_ratio, args.seq_len, args.pre_len, args.week_len, args.batch_size, args.batch_size)

    else:
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
        dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None



    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, args.fusion)

    print("start training...",flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        train_acc = []
        train_r2 = []
        train_var = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x_w, x_d, x_c, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx_w = torch.Tensor(x_w).to(device)
            trainx_w = trainx_w.transpose(1, 3)

            trainx_d = torch.Tensor(x_d).to(device)
            trainx_d = trainx_d.transpose(1, 3)

            trainx_c = torch.Tensor(x_c).to(device)
            trainx_c = trainx_c.transpose(1, 3)

            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            input = [trainx_w, trainx_d, trainx_c]
            metrics = engine.train(input, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_acc.append(metrics[3])
            train_r2.append(metrics[4])
            train_var.append(metrics[5])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train acc: {:.4f}, Train r2: {:.4f}, Train var: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1], train_acc[-1], train_r2[-1], train_var[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_acc = []
        valid_r2 = []
        valid_var =[]

        s1 = time.time()
        for iter, (x_w, x_d, x_c, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx_w = torch.Tensor(x_w).to(device)
            testx_w = testx_w.transpose(1, 3)

            testx_d = torch.Tensor(x_d).to(device)
            testx_d = testx_d.transpose(1, 3)

            testx_c = torch.Tensor(x_c).to(device)
            testx_c = testx_c.transpose(1, 3)

            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            input  = [testx_w, testx_d, testx_c]
            metrics = engine.eval(input, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_acc.append(metrics[3])
            valid_r2.append(metrics[4])
            valid_var.append(metrics[5])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_acc = np.mean(train_acc)
        mtrain_r2 = np.mean(train_r2)
        mtrain_var = np.mean(train_var)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_acc = np.mean(valid_acc)
        mvalid_r2 = np.mean(valid_r2)
        mvalid_var = np.mean(valid_var)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train acc: {:.4f}, Train r2: {:.4f}, Train var: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid acc: {:.4f}, Valid r2: {:.4f}, Valid var: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_acc, mtrain_r2, mtrain_var, mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_acc, mvalid_r2, mvalid_var, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+"_"+ str(args.fusion)+'_'+str(args.predtime)+"m.pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+'_'+str(args.fusion)+'_'+str(args.predtime)+"m.pth"))


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x_w, x_d, x_c, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx_w = torch.Tensor(x_w).to(device)
        testx_w = testx_w.transpose(1,3)

        testx_d = torch.Tensor(x_d).to(device)
        testx_d = testx_d.transpose(1, 3)

        testx_c = torch.Tensor(x_c).to(device)
        testx_c = testx_c.transpose(1, 3)

        input =[testx_w, testx_d, testx_c]
        with torch.no_grad():
            preds = engine.model(input).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    aacc = []
    ar2 = []
    avar = []
    for i in range(args.seq_len):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]

        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test acc: {:.4f}, Test r2: {:.4f}, Test var: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        aacc.append(metrics[3])
        ar2.append(metrics[4])
        avar.append(metrics[5])
    """
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape), np.mean(armse)))
    #torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+"_"+str(args.fusion)+'_'+str(args.predtime)+"m.pth")
    rmse_15, rmse_30, rmse_60 = np.mean(armse[:3]), np.mean(armse[:6]), np.mean(armse[:12])
    mae_15, mae_30, mae_60 = np.mean(amae[:3]), np.mean(amae[:6]), np.mean(amae[:12])
    mape_15, mape_30, mape_60 = np.mean(amape[:3]), np.mean(amape[:6]), np.mean(amape[:12])
    log = 'On average over 6 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(mae_30), np.mean(mape_30), np.mean(rmse_30)))
    log = 'On average over 3 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(mae_15), np.mean(mape_15), np.mean(rmse_15)))
    torch.save([engine.model.state_dict(), rmse_15, rmse_30, rmse_60, mae_15, mae_30, mae_60, mape_15, mape_30,mape_60],
               args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + "_" + str(args.fusion) +
               '_' + str(args.predtime)+ 'm_rmse_'+ str(np.round(rmse_60,4))+'_mae_'+str(np.round(mae_60,4))+'_mape_'+str(np.round(mape_60,4)) + ".pth")
    """
    log = 'On average over 4 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test acc: {:.4f}, Test r2: {:.4f}, Test var: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(aacc), np.mean(ar2), np.mean(avar)))
    # torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+"_"+str(args.fusion)+'_'+str(args.predtime)+"m.pth")
    rmse_15, rmse_30, rmse_45, rmse_60 = np.mean(armse[:1]), np.mean(armse[:2]), np.mean(armse[:3]), np.mean(armse[:4])
    mae_15, mae_30, mae_45, mae_60 = np.mean(amae[:1]), np.mean(amae[:2]), np.mean(amae[:3]), np.mean(amae[:4])
    mape_15, mape_30, mape_45, mape_60 = np.mean(amape[:1]), np.mean(amape[:2]), np.mean(amape[:3]), np.mean(amape[:4])
    acc_15, acc_30, acc_45, acc_60 = np.mean(aacc[:1]), np.mean(aacc[:2]), np.mean(aacc[:3]), np.mean(aacc[:4])
    r2_15, r2_30, r2_45, r2_60 = np.mean(ar2[:1]), np.mean(ar2[:2]), np.mean(ar2[:3]), np.mean(ar2[:4])
    var_15, var_30, var_45, var_60 = np.mean(avar[:1]), np.mean(avar[:2]), np.mean(avar[:3]), np.mean(avar[:4])
    log = 'On average over 3 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test acc: {:.4f}, Test r2: {:.4f}, Test var: {:.4f}'
    print(log.format(np.mean(mae_45), np.mean(mape_45), np.mean(rmse_45), np.mean(acc_45), np.mean(r2_45), np.mean(var_45)))
    log = 'On average over 2 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test acc: {:.4f}, Test r2: {:.4f}, Test var: {:.4f}'
    print(log.format(np.mean(mae_30), np.mean(mape_30), np.mean(rmse_30), np.mean(acc_30), np.mean(r2_30), np.mean(var_30)))
    log = 'On average over 1 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test acc: {:.4f}, Test r2: {:.4f}, Test var: {:.4f}'
    print(log.format(np.mean(mae_15), np.mean(mape_15), np.mean(rmse_15), np.mean(acc_15), np.mean(r2_15), np.mean(var_15)))
    torch.save(
        [engine.model.state_dict(), rmse_15, rmse_30, rmse_60, mae_15, mae_30, mae_60, mape_15, mape_30, mape_60],
        args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + "_" + str(args.fusion) +
        '_' + str(args.predtime) + 'm_rmse_' + str(np.round(rmse_60, 4)) + '_mae_' + str(
            np.round(mae_60, 4)) + '_mape_' + str(np.round(mape_60, 4))+ '_acc_' + str(np.round(acc_60, 4))+ '_r2_' + str(np.round(r2_60, 4))+ '_var_' + str(np.round(var_60, 4)) + ".pth")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
