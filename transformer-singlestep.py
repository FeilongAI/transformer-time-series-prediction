import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)

input_window = 100 # number of input steps
output_window = 1 # number of prediction steps, in this model its fixed to one
block_len = input_window + output_window # for one input-output pair
batch_size = 10
train_size = 0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): # d_model 模型的维度 max_len 最大序列长度，默认为5000
        # 调用父类nn.Module的初始化方法
        super(PositionalEncoding, self).__init__()
        # 创建一个形状为(max_len, d_model)的零张量,用于存储位置编码
        pe = torch.zeros(max_len, d_model) # 创建位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 创建一个列向量,包含从0到max_len-1的位置索引
        # unsqueeze(1)将其变为列向量,形状为(max_len, 1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        # 计算位置编码的频率项
        # 使用numpy的arange函数创建一个数组,然后用它来计算div_term
        # 这个计算方法是位置编码的核心,用于生成不同频率的正弦波
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        # 使用正弦函数填充偶数列
        # 0::2表示从索引0开始,步长为2,即所有偶数索引
        pe[:, 1::2] = torch.cos(position * div_term[1::2])
        # 使用余弦函数填充奇数列
        # 1::2表示从索引1开始,步长为2,即所有奇数索引
        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, d_model],so need seq-len <= 5000
        # 首先unsqueeze(0)在第0维增加一个维度,然后transpose(0,1)交换前两个维度
        # 最终pe的形状变为(max_len, 1, d_model)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)
        # 将pe注册为模块的缓冲区
        # 这样pe就不会被视为模型参数,但会随模型一起保存和加载

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)
        # x.size(0)获取输入序列的实际长度
        # self.pe[:x.size(0), :]选择对应长度的位置编码
        # repeat(1, x.shape[1], 1)将位置编码扩展到与输入的批次大小相匹配
        # 最后将位置编码加到输入x上,实现位置信息的注入
          

class TransAm(nn.Module):
    # 初始化方法,设置模型的主要参数
    # feature_size: 特征维度,默认250
    # num_layers: Transformer编码器的层数,默认1
    # dropout: dropout率,用于防止过拟合,默认0.1
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        # 调用父类(nn.Module)的初始化方法
        super(TransAm, self).__init__()
        # 设置模型类型为'Transformer'
        self.model_type = 'Transformer'
        # 创建输入嵌入层,将1维输入转换为feature_size维
        # 这一步相当于将单变量时间序列扩展到高维空间
        self.input_embedding  = nn.Linear(1,feature_size)
        # 初始化源序列掩码为None,后续会根据需要生成
        self.src_mask = None
        # 创建位置编码器,使用之前定义的PositionalEncoding类
        # 位置编码用于为序列中的每个元素添加位置信息
        self.pos_encoder = PositionalEncoding(feature_size)
        #  # 创建一个Transformer编码器层
        #         # d_model: 模型的维度,等于feature_size
        #         # nhead: 多头注意力中的头数,这里设为10
        #         # dropout: dropout率,使用传入的参数
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        # 创建完整的Transformer编码器,包含多个编码器层
        # num_layers决定了编码器层的重复次数
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # 创建解码器,将feature_size维的特征映射回1维输出
        # 这一步相当于从高维空间映射回原始的时间序列空间
        self.decoder = nn.Linear(feature_size,1)
        # 调用权重初始化方法
        self.init_weights()

    def init_weights(self):#  # 初始化模型权重的方法
        # 设置初始化范围
        initrange = 0.1
        # 将解码器的偏置初始化为0
        self.decoder.bias.data.zero_()
        # 将解码器的权重初始化为均匀分布
        # 范围是[-initrange, initrange]
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        # 定义模型的前向传播方法
        # src: 输入数据,形状为(input_window, batch_size, 1)
        # 如果源序列掩码未创建或大小不匹配
        if self.src_mask is None or self.src_mask.size(0) != len(src):#torch.Size([100, 10, 1])
            # 获取输入数据所在的设备(CPU或GPU)
            device = src.device
            # 生成适当大小的掩码并移到相同设备
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        # 输入嵌入
        src = self.input_embedding(src) # linear transformation before positional embedding
        # 添加位置编码 torch.Size([100, 10, 250])
        src = self.pos_encoder(src)
        # 通过Transformer编码器
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        # 解码得到最终输出
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        # 生成上三角矩阵
        # 创建上三角矩阵,然后转置
        # 这确保了每个位置只能注意到它自己和之前的位置
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

        # 将掩码转换为浮点数
        # 将0替换为负无穷(表示屏蔽),将1替换为0.0(表示保留)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
'''
In fact, assuming that the number of samples is N, 
the length of the input sequence is m, and the backward prediction is k steps, 
then length of a block [input : 1 , 2 ... m  -> output : k , k+1....m+k ] 
should be (m+k) :  block_len, so to ensure that each block is complete, 
the end element of the last block should be the end element of the entire sequence, 
so the actual number of blocks is [N - block_len + 1] 
'''
def create_inout_sequences(input_data, input_window ,output_window):
    inout_seq = []
    L = len(input_data)
    block_num =  L - block_len + 1 #总的长度3200 -101
    # total of [N - block_len + 1] blocks
    # where block_len = input_window + output_window

    for i in range( block_num ):
        train_seq = input_data[i : i + input_window]#0 100
        train_label = input_data[i + output_window : i + input_window + output_window]#1，101 最后一个应该是预测值
        inout_seq.append((train_seq ,train_label))

    return torch.FloatTensor(np.array(inout_seq))

def get_data():
    # construct a littel toy dataset
    # np.arange(start, stop, step) 生成一个数组
    # 从0开始，到400结束（不包括400），步长为0.1
    # 这将生成4000个数据点 (400 / 0.1 = 4000)
    time        = np.arange(0, 400, 0.1) #生成4000个数据点
    # 1. np.sin(time): 基本的正弦波
    # 2. np.sin(time * 0.05): 频率较低的正弦波
    # 3. np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time)):
    #    - 频率介于前两者之间的正弦波
    #    - 乘以一个均值为-0.2，标准差为0.2的正态分布随机数
    #    - 这部分添加了随机噪声，使信号更接近真实世界的数据
    amplitude   = np.sin(time) + np.sin(time * 0.05) + \
                  np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))#生成了一个复合的时间序列信号


    
    #loading weather data from a file
    #from pandas import read_csv
    #series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    
    # looks like normalizing input values curtial for the model
    #数据进行归一化处理 MinMaxScaler期望输入是一个二维数组，其中每列代表一个特征
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    #amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    #fit_transform方法同时执行了两个操作：
    # fit: 计算用于缩放的最小值和最大值。
     #transform: 使用计算出的参数对数据进行实际的缩放。
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    #训练数据集大小
    sampels = int(len(time) * train_size) # use a parameter to control training size
    train_data = amplitude[:sampels]
    #切割训练数据集
    test_data = amplitude[sampels:]
    #切割测试训练集
    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)

    train_sequence = create_inout_sequences( train_data,input_window ,output_window)
    '''
    train_sequence = train_sequence[:-output_window] # todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack..
    # looks like maybe solved
    '''
    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window,output_window)
    '''
    test_data = test_data[:-output_window] # todo: fix hack?
    '''
    # shape with (block , sql_len , 2 )
    return train_sequence.to(device),test_data.to(device)


def get_batch(input_data, i , batch_size):

    # batch_len = min(batch_size, len(input_data) - 1 - i) #  # Now len-1 is not necessary
    batch_len = min(batch_size, len(input_data) - i)
    data = input_data[ i:i + batch_len ]#torch.Size([10, 2, 100])

    # stack函数将一系列张量沿着一个新维度进行堆叠。

    # view方法重塑张量的维度。
    # 1. [item[0] for item in data] 是一个列表推导，选择每个数据项的第一个元素
    # 2. torch.stack() 将这些元素堆叠成一个新的张量
    # 3. .view((input_window, batch_len, 1)) 重塑张量维度为 (序列长度, 批次大小, 特征数)
    # 这里假设 input_window 是预定义的全局变量，表示输入序列的长度
    input = torch.stack([item[0] for item in data]).view((input_window,batch_len,1))
    # ( seq_len, batch, 1 ) , 1 is feature size
    target = torch.stack([item[1] for item in data]).view((input_window,batch_len,1))
    return input, target

def train(train_data):
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data), batch_size)):  # Now len-1 is not necessary
        # data and target are the same shape with (input_window,batch_len,1)
        data, targets = get_batch(train_data, i , batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        # for i in range(0, len(data_source) - 1):
        for i in range(len(data_source)):  # Now len-1 is not necessary
            data, target = get_batch(data_source, i , 1) # one-step forecast
            output = eval_model(data)            
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    #test_result = test_result.cpu().numpy() -> no need to detach stuff.. 
    len(test_result)

    pyplot.plot(test_result,color="red")
    pyplot.plot(truth[:500],color="blue")
    pyplot.plot(test_result-truth,color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
    pyplot.close()
    return total_loss / i


# predict the next n steps based on the input data 
def predict_future(eval_model, data_source,steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source , 0 , 1)
    with torch.no_grad():
        for i in range(0, steps):            
            output = eval_model(data[-input_window:])
            # (seq-len , batch-size , features-num)
            # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
            data = torch.cat((data, output[-1:])) # [m,m+1,..., m+n+1]

    data = data.cpu().view(-1)
    
    # I used this plot to visualize if the model pics up any long therm structure within the data.
    pyplot.plot(data,color="red")       
    pyplot.plot(data[:input_window],color="blue")    
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-future%d.png'%steps)
    pyplot.show()
    pyplot.close()
        

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        # for i in range(0, len(data_source) - 1, eval_batch_size): # Now len-1 is not necessary
        for i in range(0, len(data_source), eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

train_data, val_data = get_data()
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005 
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    if ( epoch % 5 == 0 ):
        val_loss = plot_and_loss(model, val_data,epoch)
        predict_future(model, val_data,200)
    else:
        val_loss = evaluate(model, val_data)
   
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    #if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step() 

#src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number) 
#out = model(src)
#
#print(out)
#print(out.shape)
