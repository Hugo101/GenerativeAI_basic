import torch
import torch.nn as nn
import torch.optim as optim
import math 
import torch.utils.data as data
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model = 512, dropout = 0.1, max_len = 5000):
        '''
        d_model: dimension of the word embedding, 512
        max_len: the maximum number of tokens in a sentence, 5000
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Generate a max_len * d_model matrix, that is, 5000 * 512
        # 5000 is the maximum number of tokens in a sentence, 
        # and 512 is the length of a token represented by a vector.
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # pos：[max_len,1]
        # First calculate the fraction in the brackets, pos is [5000,1], the denominator is [256],
        # and the result of multiplication by broadcasting is [5000,256]
        div_term = pos / pow(10000.0,torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        # A sentence needs to do pe once, and there will be multiple sentences in a batch,
        # so add one dimension to do broadcast when adding to a batch of input data
        pe = pe.unsqueeze(0) # [5000,512] -> [1,5000,512] 
        # English: register_buffer is used to save the parameters that will not be updated, and the parameters are saved in the buffer
        self.register_buffer('pe', pe)
        
        
    def forward(self, x):
        '''x: [batch_size, seq_len, d_model]'''
        # 5000 is the maximum seq_len we have defined, that is to say, we have calculated the pe for the most situation
        x = x + self.pe[:, :x.size(1), :]  # Note: residual connection
        return self.dropout(x) 
        # return: [batch_size, seq_len, d_model]


# The sentences we input into the model vary in length, and we use a placeholder 'P' to pad them to the length of the longest sentence. These placeholders are meaningless and we set these positions to True. The following function returns a Boolean tensor indicating whether the position is a placeholder.
# Return: tensor [batch_size, len_q, len_k]，True means the position is a placeholder

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, len_q]
    seq_k: [batch_size, len_k]
    '''
    batch_size, len_q = seq_q.size()
    _,          len_k = seq_k.size()
    # seq_k.data.eq(0):，element in seq_k will be True (if ele == 0), False otherwise.
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # pad_attn_mask: [batch_size,1,len_k]

    # To provide a k for each q, so the second dimension is expanded q times.
    # Expand is not really doubling the memory, it just repeats the reference, and any modification to any reference will modify the original value.
    # Here we use it to save memory because we won't modify this mask.
    return pad_attn_mask.expand(batch_size, len_q, len_k) # return: [batch_size, len_q, len_k]
    # Return batch_size len_q * len_k matrix, content is True and False, True means the position is a placeholder.
    # The i-th row and the j-th column indicate whether the attention of the i-th word of the query to the j-th word of the key is meaningless. If it is meaningless, it is True. If it is meaningful, it is False (that is, the position of the padding is True)


# To prevent positions from attending to subsequent positions
def get_attn_subsequence_mask(seq):
    """
    seq: [batch_size, tgt_len]
    This is only used in decoder, so the length of seq is the length of target sentence.
    """
    batch_size, tgt_len = seq.shape
    attn_shape = [batch_size, tgt_len, tgt_len]
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    # np.triu is to generate an upper triangular matrix, k is the offset relative to the main diagonal
    # k = 1 means not including the main diagonal (starting from the main diagonal offset 1
    # subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    # subsequence_mask = torch.from_numpy(subsequence_mask).byte() 
    subsequence_mask = torch.triu(torch.ones(attn_shape), diagonal=1).byte() #.byte() is equivalent to .to(torch.uint8)
    # Because there are only 0 and 1, byte is used to save memory.
    return subsequence_mask  # return: [batch_size, tgt_len, tgt_len]




class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]  
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v] 
        Two types of attention:
        1) self attention
        2) cross attention: K and V are encoder's output, so the shape of K and V are the same
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        
        # len_q is not necessary to be euqal to len_k
        because len_q is the length of the query sentence (decoder input, or predicted sentence in the 2) attention operation),
        and len_k is the length of the key sentence (encoder input, or source sentence).
        '''
        batch_size, n_heads, len_q, d_k = Q.shape 
        # 1) computer attention score QK^T/sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores: [batch_size, n_heads, len_q, len_k]
        # 2) mask operation (option), only used in the decoder
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, float('-inf')) # or float('-inf'), -1e9: negative infinity
            # Fills elements of self tensor with value where mask is True.
            # The masked elements in the scores are replaced by -1e9, 
            # so that the softmax operation will make the value of the masked position close to 0.

        # 3) softmax to get attention weights
        attn = nn.Softmax(dim=-1)(scores)  # attn: [batch_size, n_heads, len_q, len_k]
        # 4) use attention weights to weigh the value V
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        '''
        返回的context: [batch_size, n_heads, len_q, d_v]本质上还是batch_size个句子，
        只不过每个句子中词向量维度512被分成了8个部分，分别由8个头各自看一部分，每个头算的是整个句子(一列)的512/8=64个维度，最后按列拼接起来
        '''
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 512, n_heads = 8, dropout_rate = 0.0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.head_dim = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, len_q, d_model]
        K: [batch_size, len_k, d_model]
        V: [batch_size, len_v, d_model] 
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size, len_q, d_model = Q.shape
        batch_size, len_k, d_model = K.shape
        batch_size, len_v, d_model = V.shape
        
        # 1) linear projection
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # 2) split by heads
        # [batch_size, len_q, d_model] -> [batch_size, len_q, n_heads, head_dim]
        Q = Q.reshape(batch_size, len_q, self.n_heads, self.head_dim)
        K = K.reshape(batch_size, len_k, self.n_heads, self.head_dim)
        V = V.reshape(batch_size, len_v, self.n_heads, self.head_dim)
        
        # 3) transpose for attention dot product
        # [batch_size, len_q, n_heads, head_dim] -> [batch_size, n_heads, len_q, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 4) attention
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn_mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        context = self.scaled_dot_product_attention(Q, K, V, attn_mask)
        # context: [batch_size, n_heads, len_q, head_dim]
        
        # 5) concat heads
        # method 1:
        output = context.transpose(1, 2).reshape(batch_size, len_q, self.d_model)
        # output: [batch_size, len_q, d_model]
        
        # method 2:
        # output = torch.cat([context[:,i,:,:] for i in range(self.n_heads)], dim=-1)
        # output: [batch_size, len_q, d_model]
        
        # 6) linear projection (concat heads)
        output = self.W_O(output)
        return output # output: [batch_size, len_q, d_model]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, 
                 d_model = 512, 
                 d_ff = 2048, 
                 dropout_rate = 0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff # dimension of latent layer for feed forward neural network
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        '''
        output = self.relu(self.W_1(x))
        output = self.W_2(output)
        
        return output


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class EncoderLayer(nn.Module):
    def __init__(self, 
                 d_model = 512, 
                 d_ff = 2048, 
                 n_heads = 8, 
                 dropout_rate = 0.0):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs:         [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # Sublayer 1: self attention
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size, src_len, d_model]
        
        # add & norm
        enc_outputs = self.layer_norm1(enc_inputs + enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        
        # Sublayer 2: position-wise feed forward network
        enc_ff_outputs = self.pos_ffn(enc_outputs)
        # enc_ff_outputs: [batch_size, src_len, d_model]
        
        # add & norm
        enc_outputs = self.layer_norm2(enc_outputs + enc_ff_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        
        return enc_outputs


class Encoder(nn.Module):
    def __init__(
        self,
        d_model = 512, 
        d_ff = 2048, 
        n_heads = 8, 
        n_layers = 6,
        dropout_rate = 0.0, 
        ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, dropout_rate) for _ in range(n_layers)])
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = enc_inputs
        # encoding
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        # enc_outputs: [batch_size, src_len, d_model]
        
        return enc_outputs


class DecoderLayer(nn.Module):
    def __init__(self, 
                 d_model = 512, 
                 d_ff = 2048, 
                 n_heads = 8, 
                 dropout_rate = 0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.dec_cros_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.layer_norm_3 = LayerNorm(d_model)
        
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs:         [batch_size, tgt_len, d_model]
        enc_outputs:        [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask:  [batch_size, tgt_len, src_len]
        '''
        # Sublayer 1: self attention
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        
        # add & norm
        dec_outputs = self.layer_norm_1(dec_inputs + dec_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model]
        
        # Sublayer 2: encoder-decoder cross attention, 
        # Q (decoder), K (encoder output), V (encoder output)
        dec_outputs_2 = self.dec_cros_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        
        # add & norm
        dec_outputs_2 = self.layer_norm_2(dec_outputs + dec_outputs_2)
        # dec_outputs_2: [batch_size, tgt_len, d_model]
        
        # Sublayer 3: position-wise feed forward network
        dec_outputs_3 = self.pos_ffn(dec_outputs_2)
        # dec_ff_outputs: [batch_size, tgt_len, d_model]
        
        # add & norm
        dec_outputs_3 = self.layer_norm_3(dec_outputs_2 + dec_outputs_3)
        # dec_outputs_3: [batch_size, tgt_len, d_model]
        
        return dec_outputs_3 


class Decoder(nn.Module):
    def __init__(self, 
                 d_model = 512, 
                 d_ff = 2048, 
                 n_heads = 8, 
                 n_layers = 6,
                 dropout_rate = 0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, dropout_rate) for _ in range(n_layers)])
    
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_self_attn_subsequence_mask, memory_self_attn_mask):
        '''
        dec_inputs:         [batch_size, tgt_len]
        enc_outputs:        [batch_size, src_len, d_model]
        sometimes, enc_outputs is also called memory in the paper
        '''
        # combine two masks in decoder self attention
        dec_self_attn_mask = torch.gt((dec_self_attn_mask + dec_self_attn_subsequence_mask), 0)
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        
        output = dec_inputs
        # decoding
        for layer in self.layers:
            output = layer(output, enc_outputs, dec_self_attn_mask, memory_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        
        return output


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens): # tokens: [batch_size, seq_len]
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# The Transformer model
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model = 512, 
                 d_ff = 2048, 
                 n_heads = 8, 
                 n_layers = 6,
                 dropout_rate = 0.1):
        super(Transformer, self).__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model) #[batch_size, src_len, d_model]
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model) #[batch_size, tgt_len, d_model]
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate)

        self.Encoder = Encoder(d_model, d_ff, n_heads, n_layers, dropout_rate)
        self.Decoder = Decoder(d_model, d_ff, n_heads, n_layers, dropout_rate)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
    def create_masks(self, src, tgt):
        '''
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
        
        ''' 
        # padding mask for encoder self attention: enc_self_attn_pad_mask
        src_key_padding_mask = get_attn_pad_mask(src, src) #[batch_size, src_len, src_len]
        
        # padding mask for decoder self attention: dec_self_attn_pad_mask
        tgt_key_padding_mask = get_attn_pad_mask(tgt, tgt) # [batch_size, tgt_len, tgt_len]
        
        # encoder-decoder cross attention mask: cross_attn_mask
        memory_key_padding_mask = get_attn_pad_mask(tgt, src) #[batch_size, tgt_len, src_len]
        
        # sequence mask (only exists in decoder)
        tgt_mask = get_attn_subsequence_mask(tgt) # [batch_size, tgt_len, tgt_len]
        
        if tgt_key_padding_mask.is_cuda:
            tgt_mask = tgt_mask.cuda()
        
        return  tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask


    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # embedding + position encoding
        src_emb = self.pos_embedding(self.src_tok_emb(enc_inputs))
        tgt_emb = self.pos_embedding(self.tgt_tok_emb(dec_inputs))
        
        # prepare masks
        tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = self.create_masks(enc_inputs, dec_inputs)
        
        # encoding
        enc_outputs = self.Encoder(src_emb, src_key_padding_mask)
        # enc_outputs: [batch_size, src_len, d_model]
        
        # decoding
        dec_outputs = self.Decoder(tgt_emb, enc_outputs, tgt_key_padding_mask, tgt_mask, memory_key_padding_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        
        # projection
        dec_logits = self.generator(dec_outputs)
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        
        return dec_logits.view(-1, dec_logits.size(-1))  #  [batch_size * tgt_len, tgt_vocab_size]
        # there are batch_size sentences in one batch
        # first tgt_len words are the prediction probability of the first sentence,
        # then the next tgt_len words are the prediction probability of the second sentence, and so on.
