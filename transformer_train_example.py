import torch
import torch.nn as nn
import torch.optim as optim
import math 
import torch.utils.data as data
import numpy as np
from transformer import *

# Dataset preparation
# S: Start of sentence
# E: End of sentence
# P: padding，make sure the length of the sentence is the same

sentence = [
    # enc_input   dec_input    dec_output
    ['ich mochte ein bier P','S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P','S i want a coke .', 'i want a coke . E'],
]
# source vocab
src_vocab = {'P':0, 'ich':1,'mochte':2,'ein':3,'bier':4,'cola':5}
src_vocab_size = len(src_vocab) # 6

# target vocab (including special symbols)
tgt_vocab = {'P':0,'i':1,'want':2,'a':3,'beer':4,'coke':5,'S':6,'E':7,'.':8}

# reverse mapping dictionary, idx ——> word
idx2word = {v:k for k,v in tgt_vocab.items()}
tgt_vocab_size = len(tgt_vocab) # 9

src_len = 5
# the length of the longest sentence in the input sequence, 
# which is actually the number of tokens in the longest sentence
tgt_len = 6
# the length of the longest sentence in the dec_input/dec_output sequence


# 这个函数把原始输入序列转换成token表示
def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [],[],[]
    for i in range(len(sentence)):
        enc_input = [src_vocab[word] for word in sentence[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentence[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentence[i][2].split()]
        
        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)
        
    # LongTensor是专用于存储整型的，Tensor则可以存浮点、整数、bool等多种类型
    return torch.LongTensor(enc_inputs),torch.LongTensor(dec_inputs),torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentence)

print(' enc_inputs: \n', enc_inputs)  # enc_inputs: [2,5]
print(' dec_inputs: \n', dec_inputs)  # dec_inputs: [2,6]
print(' dec_outputs: \n', dec_outputs) # dec_outputs: [2,6]


class MyDataSet(data.Dataset):
    def __init__(self,enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet,self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        
    def __len__(self):
        # in the example above, enc_inputs.shape = [2,5], so return 2
        return self.enc_inputs.shape[0] 
    
    # return a set of enc_input, dec_input, dec_output, according to the index
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# DataLoader
loader = data.DataLoader(dataset=MyDataSet(enc_inputs,dec_inputs, dec_outputs),batch_size=2,shuffle=True)
len(loader)



model = Transformer(src_vocab_size, 
                    tgt_vocab_size, 
                    d_model = 512, 
                    d_ff = 2048, 
                    n_heads = 8, 
                    n_layers = 6,
                    dropout_rate = 0.1).cuda()
model.train()
# 损失函数,忽略为0的类别不对其计算loss（因为是padding无意义）
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# training
num_epochs = 10
for epoch in range(num_epochs):
    for enc_inputs, dec_inputs, dec_outputs_true in loader:
        '''
        enc_inputs: [batch_size, src_len] [2,5]
        dec_inputs: [batch_size, tgt_len] [2,6]
        dec_outputs_true: [batch_size, tgt_len] [2,6]
        '''
        enc_inputs, dec_inputs, dec_outputs_true = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs_true.cuda()
        outputs = model(enc_inputs, dec_inputs) # outputs: [batch_size * tgt_len, tgt_vocab_size]

        print("pred: ", outputs.shape, outputs)
        print("true: ", dec_outputs_true.view(-1).shape, dec_outputs_true.view(-1))
        loss = criterion(outputs, dec_outputs_true.view(-1))  # 将dec_outputs展平成一维张量

        # weight updates
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        break

torch.save(model, 'MyTransformer.pth')

