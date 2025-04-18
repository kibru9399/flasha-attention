import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q: tl.constexpr,
            BLOCK_SIZE_KV: tl.constexpr,
            STAGE: tl.constexpr,
            offs_q: tl.constexpr,
            offs_kv: tl.constexpr,
            SEQ_LEN: tl.constexpr,
):
    #range of values handled by this stage
    if STAGE == 1:
        # from 0 to the left of the diagonal
        #lo, high tell us what is lower and higher index of the key block that this particular stage 
        #should be working with
        lo, hi = 0, block_index_q*BLOCK_SIZE_Q
    elif STAGE == 2:
        #used for the block in which there is a transition between masked and non-masked keys
        lo, hi = block_index_q*BLOCK_SIZE_Q, (block_index_q + 1)*BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        lo, hi = 0, SEQ_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr,(lo, 0))
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block*softmax_scale + tl.where(mask, 0, 1e-6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1)*softmax_scale)
            QK_block = QK_block*softmax_scale - m_ij[:, None]

        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i*alpha + l_ij
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        O_block = O_block*alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) #seq_len, head_dim
        K_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_SIZE_KV)) #K[HEAD_DIM, SEQ_LEN]

    return O_block, l_i, m_i



@triton.jit
def _attn_fwd(
    Q,
    K, 
    V, 
    softmax_scale,
    M, 
    O, 
    stride_Q_batch, 
    stride_Q_head, 
    stride_Q_seq, 
    stride_Q_dim,
    stride_K_batch, 
    stride_K_head, 
    stride_K_seq, 
    stride_K_dim,
    stride_V_batch, 
    stride_V_head, 
    stride_V_seq, 
    stride_V_dim,
    stride_O_batch, 
    stride_O_head, 
    stride_O_seq, 
    stride_O_dim,
    BATCH_SIZE, 
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr, 
    STAGE: tl.constexpr

):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head//NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qvk_offset = (
        index_batch.to(tl.int64)*stride_Q_batch
        + index_batch_head.to(tl.int64)*stride_Q_head
    )
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, #Q[index_batch, index_head, block_index_q*BLOCK_SIZE_Q:, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offset=(block_index_q*BLOCK_SIZE_Q, 0), 
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, #Q[index_batch, index_head, :, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offset=(0, 0), 
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, #k[index_batch, index_head, :, :]
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offset=(0, 0), 
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset, #Q[index_batch, index_head, block_index_q*BLOCK_SIZE_Q:, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offset=(block_index_q*BLOCK_SIZE_Q, 0), 
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    #the offset for the tokens in Q to process
    offs_q = block_index_q*BLOCK_SIZE_Q + tl.range(0, BLOCK_SIZE_Q)

    #the offsets for  the tokens in the K AND V to process
    offs_kv = tl.range(0, BLOCK_SIZE_KV)
    #m_i: the running maximum, we have on for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')
    #l_i: the running sum we have one for each query
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    #the accumulator for the output, group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    #load the blocks of Q: it will stay in the SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )


class TritonAttention(torch.autugrad.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1],K.shape[-1], V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.ones_like(Q)
        stage = 3 if causal else 1
        #this code tells us how many programs are there that can be parallelized and it would be upto the
        #GPU to decide how many of them would ACTUALLY run in parallel based on available resource
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), #which block of seqeunce of queries
            BATCH_SIZE*NUM_HEADS, #Which head of which batch
            1   #Z dimension in cuda launch grid

        )
        #number of parallel programs: BATCH_SIZE*NUM_HEADS*NUM_BLOCKS_Q

        #M is the logsumexp for the backward pass , one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=Q.device, dtype=torch.float32
        )
        _attn_fwd[grid](
            Q=Q, 
            K=K, 
            V=V, 
            softmax_scale=softmax_scale, 
            M=M, 
            O=O, 
            stride_Q_batch=Q.stride[0], 
            stride_Q_head=Q.stride[1], 
            stride_Q_seq=Q.stride[2], 
            stride_Q_dim=Q.stride[3],
            stride_K_batch=K.stride[0], 
            stride_K_head=K.stride[1], 
            stride_K_seq=K.stride[2], 
            stride_K_dim=K.stride[3],
            stride_V_batch=V.stride[0], 
            stride_V_head=V.stride[1], 
            stride_V_seq=V.stride[2], 
            stride_V_dim=V.stride[3],
            stride_O_batch=O.stride[0], 
            stride_O_head=O.stride[1], 
            stride_O_seq=O.stride[2], 
            stride_O_dim=O.stride[3],
            BATCH_SIZE=BATCH_SIZE, 
            SEQ_LEN=SEQ_LEN, 
            HEAD_DIM=HEAD_DIM_K, 
            STAGE=stage

        )
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return D

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda')
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda')
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda')
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    softmax_scale = 1/(HEAD_DIM**0.5)
    dO = torch.randn_like(Q)
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device='cuda'))
    P = torch.matmul(Q, K.transpose(2, 3))*softmax_scale
    if causal:
        P[:, :, MASK == 0] = float('-inf')
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dv, v.gad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None 

    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dv, v.gad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None 
    
    #compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
