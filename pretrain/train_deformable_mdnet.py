#coding=utf-8
import os
import sys
import pickle
import time

from data_prov import *
from model import *
from options import *
import mxnet as mx
from mxnet import nd
import os
import scipy.io
import numpy as np
np.set_printoptions(precision=3)


img_home = '../dataset/'
data_path = 'data/vot-otb.pkl'
optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': 0.0001,
                        'clip_gradient': 10}
init_model_path = '../models/imagenet-vgg-m.mat'

def precision(pos_score,neg_score):
    scores = nd.concat(pos_score[:,1],neg_score[:,1],dim=0)
    topk = nd.topk(scores,k=32)
    prec = (nd.sum((topk < 32).astype('float32')) / 32+1e-8)
    return prec

def load_model_from_file(model_path):
    mat = scipy.io.loadmat(model_path)
    mat_layers = list(mat['layers'])[0]
    weights=[]
    biaes=[]
    for i in range(3):
        weight, bias = mat_layers[i*4]['weights'].item()[0]
        weights.append(nd.array(np.transpose(weight,(3,2,0,1))))
        biaes.append(nd.array(bias[:,0]))
    arg_params={'conv1_weight':weights[0],'conv2_weight':weights[1],'conv3_weight':weights[2],
                'conv1_bias':biaes[0],'conv2_bias':biaes[1],'conv3_bias':biaes[2]}
    return arg_params

def get_mdnet_symbol(K):
    data = mx.sym.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, name='conv1', num_filter=96,
                                kernel=(7,7), stride=(2,2))
    conv1_relu = mx.symbol.Activation(name='conv1_relu', data=conv1, act_type='relu')
    conv1_lrn = mx.symbol.LRN(data=conv1_relu, alpha=0.0001, beta=0.75, knorm=2, nsize=5, name='conv1_lrn')
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1_lrn, kernel=(3,3),
                            stride=(2,2), pool_type='max')
    conv2 = mx.symbol.Convolution(data = pool1, name='conv2', num_filter=256,
                                kernel=(5,5), stride=(2,2))
    conv2_relu = mx.symbol.Activation(name='conv2_relu', data=conv2, act_type='relu')
    conv2_lrn = mx.symbol.LRN(data=conv2_relu, alpha=0.0001, beta=0.75, knorm=2, nsize=5, name='conv2_lrn')
    pool2 = mx.symbol.Pooling(name='pool2', data=conv2_lrn, kernel=(3,3),
                            stride=(2,2), pool_type='max')
    conv3 = mx.symbol.Convolution(data = pool2, name='conv3', num_filter=512,
                                kernel=(3,3), stride=(1,1))
    conv3_relu = mx.symbol.Activation(name='conv3_relu', data=conv3, act_type='relu')
    conv3_relu_reshape = mx.symbol.reshape(data=conv3_relu, shape=(0,-1), name='conv3_relu_reshape')
    fc4_drop_out = mx.symbol.Dropout(data=conv3_relu_reshape,p=0.5, name='fc4_drop_out')
    fc4_weight = mx.symbol.Variable('fc4_weight', lr_mult=10)
    fc4 = mx.symbol.FullyConnected(data=fc4_drop_out, num_hidden=512, name='fc4',weight=fc4_weight)
    fc4_relu = mx.symbol.Activation(data=fc4,act_type='relu',name='fc4_relu')
    fc5_drop_out = mx.symbol.Dropout(data=fc4_relu, p=0.5,name='fc5_drop_out')
    fc5_weight = mx.symbol.Variable('fc5_weight', lr_mult=10)
    fc5 = mx.symbol.FullyConnected(data=fc5_drop_out, num_hidden=512, name='fc5', weight=fc5_weight)
    fc5_relu = mx.symbol.Activation(data=fc5,act_type='relu',name='fc5_relu')

    binary_losses = []
    for i in range(K):
        fc6_drop_out = mx.symbol.Dropout(data=fc5_relu,name='fc6_'+str(i)+'_drop_out',p=0.5)
        fc6_weight = mx.symbol.Variable('fc6_'+str(i)+'_weight', lr_mult=10)
        fc6 = mx.symbol.FullyConnected(data=fc6_drop_out, name='fc6_'+str(i), num_hidden=2,weight=fc6_weight)
        fc6_stop_grad = mx.symbol.BlockGrad(fc6)
        #print(fc6.list_outputs())
        '''binary_loss_stop_grad = mx.symbol.BlockGrad(mx.symbol.sum(mx.symbol.slice_axis(-mx.symbol.log_softmax(mx.symbol.slice_axis(fc6,axis=0,begin=0,end=32)),
                                                                                axis=1,begin=1,end=2))
                                                +
                                                mx.symbol.sum(mx.symbol.slice_axis(-mx.symbol.log_softmax(mx.symbol.slice_axis(fc6,axis=0,begin=32,end=128)),
                                                                                axis=1,begin=0,end=1)))'''
        #binary_loss_stop_grad = mx.symbol.sum(mx.symbol.slice_axis(-mx.symbol.log_softmax(mx.symbol.slice_axis(fc6,axis=0,begin=0,end=32)),axis=1,begin=1,end=2))+mx.symbol.sum(mx.symbol.slice_axis(-mx.symbol.log_softmax(mx.symbol.slice_axis(fc6,axis=0,begin=32,end=128)),axis=1,begin=0,end=1))
        
        binary_loss = mx.symbol.MakeLoss(mx.symbol.sum(mx.symbol.slice_axis(-mx.symbol.log_softmax(mx.symbol.slice_axis(fc6,axis=0,begin=0,end=32)),
                                                                                axis=1,begin=1,end=2))
                                                +
                                                mx.symbol.sum(mx.symbol.slice_axis(-mx.symbol.log_softmax(mx.symbol.slice_axis(fc6,axis=0,begin=32,end=128)),
                                                                                axis=1,begin=0,end=1)))
        binary_losses.append(fc6_stop_grad)
        binary_losses.append(binary_loss)
    return binary_losses

if __name__=='__main__':
    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
    K = len(data)
    dataset = [None]*K
    for k, (seqname, seq) in enumerate(data.iteritems()):
        img_list = seq['images']
        gt = seq['gt']
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = RegionDataset(img_dir, img_list, gt, opts)

    binary_losses = get_mdnet_symbol(K)
    # 保存fc6层参数
    fc6_params = [{}]*K
    # 初始化模型
    mods = []
    for i in range(len(binary_losses)/2):
        mod = mx.mod.Module(mx.symbol.Group([binary_losses[i*2],binary_losses[i*2+1]]))
        mods.append(mod)

    for i in range(50):
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        for j,k in enumerate(k_list):
            tic = time.time()
            #准备数据
            pos_regions, neg_regions = dataset[k].next()
            pos_regions = pos_regions.numpy()
            neg_regions = neg_regions.numpy()
            # 检查输入形状
            while(pos_regions.shape[0]!=32 or neg_regions.shape[0]!=96):
                pos_regions,neg_regions=dataset[k].next()
                pos_regions = pos_regions.numpy()
                neg_regions = neg_regions.numpy()
            pos_data = nd.array(pos_regions)
            neg_data = nd.array(neg_regions)
            nd_data = nd.concat(pos_data,neg_data,dim=0) 
            data = mx.io.DataBatch([nd_data]) 
            
            mod = mods[k]
            if(not mod.binded):
                mod.bind(data_shapes=[('data', nd_data.shape)])
            if(i == 0):
                if(j == 0):
                    mod.init_params(arg_params=load_model_from_file(init_model_path),aux_params=None,allow_missing=True)
                else:
                    mod.init_params(arg_params=shared_params,aux_params=None,allow_missing=True)
            else:
                mod.set_params(arg_params=dict(shared_params.items()+fc6_params[k].items()),aux_params=None)

            if(not mod.optimizer_initialized):
                mod.init_optimizer(optimizer='sgd', optimizer_params=optimizer_params)
            # 训练模型
            mod.forward(data)
            pos_score = nd.slice_axis(mod.get_outputs()[0],axis=0,begin=0,end=32)
            neg_score = nd.slice_axis(mod.get_outputs()[0],axis=0,begin=32,end=128)
            mod.backward()
            mod.update()

            
            shared_params = {'conv1_weight':mod.get_params()[0]['conv1_weight'],
                            'conv1_bias':mod.get_params()[0]['conv1_bias'],
                            'conv2_weight':mod.get_params()[0]['conv2_weight'],
                            'conv2_bias':mod.get_params()[0]['conv2_bias'],
                            'conv3_weight':mod.get_params()[0]['conv3_weight'],
                            'conv3_bias':mod.get_params()[0]['conv3_bias'],
                            'fc4_weight':mod.get_params()[0]['fc4_weight'],
                            'fc4_bias':mod.get_params()[0]['fc4_bias'],
                            'fc5_weight':mod.get_params()[0]['fc5_weight'],
                            'fc5_bias':mod.get_params()[0]['fc5_bias']}
            
            fc6_param = {'fc6_'+str(k)+'_weight':mod.get_params()[0]['fc6_'+str(k)+'_weight'],
                        'fc6_'+str(k)+'_bias':mod.get_params()[0]['fc6_'+str(k)+'_bias']}
            fc6_params[k]=fc6_param
            
            prec[k]=float(precision(pos_score,neg_score).asnumpy())
            toc = time.time()-tic
            print "Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f"%(i,j,k,float(mod.get_outputs()[1].asnumpy()),prec[k],toc)
        cur_prec = prec.mean()
        print "Mean Precision: %.3f" % (cur_prec)