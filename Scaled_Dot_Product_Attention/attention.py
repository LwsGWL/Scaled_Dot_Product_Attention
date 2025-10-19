import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) # set softmax 

    def forward(self, q, k, v, mask=None):
        # q or k or v size format: [batch_size, head, length, d_t(dim_tensor)]
        _, _, _, d_t = k.size()

        # k.transpose(2, 3) format is [batch_size, head, d_t(dim_tensor), length]
        # k.transpose(2, 3) means the transpose of k, and the reason for transposing is that 
        # the inner product of Q and k.transpose(2, 3) must be in the format [batch_size, head, lenght, lenght].
        # Formulas in the "Attention is All you Need" paper
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_t)

        # <pad> tokens are not to be counted
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)
        
        # Get Attention (Q, K, V)
        context_vector = torch.matmul(v, score)

        # The reason there are two return values ​​is because 
        # v is used for the actual calculation in Multi-Head Attention, 
        # and score is a value that can indicate which part was focused on (debugging).
        return context_vector, score

