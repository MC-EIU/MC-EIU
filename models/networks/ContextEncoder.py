import torch.nn as nn
import torch
import os
import json
import numpy as np


class ConversationalContextEncoder(nn.Module):
    """ Conversational Context Encoder """

    def __init__(self, preprocess_config=None, model_config=None):
        super(ConversationalContextEncoder, self).__init__()
        d_model = model_config.hidden_size      # ["transformer"]["encoder_hidden"]     # 注意力层的隐藏层大小
        d_cont_enc = model_config.hidden_size       # ["history_encoder"]["context_hidden"]  # 上下文编码器隐藏层大小？
        num_layers = model_config.ContextEncoder_layers     # ["history_encoder"]["context_layer"]  # 上下文编码器层数
        dropout = model_config.ContextEncoder_dropout       # ["history_encoder"]["context_dropout"]  # 上下文编码器dropout
        self.text_emb_size = model_config.input_dim_l       # ["history_encoder"]["text_emb_size"]  # 文本embedding大小
        self.visual_emb_size = model_config.input_dim_v       # ["history_encoder"]["visual_emb_size"]  # 文本embedding大小
        self.audio_emb_size = model_config.input_dim_a       # ["history_encoder"]["audio_emb_size"]  # 文本embedding大小
        self.max_history_len = model_config.ContextEncoder_max_history_len  # ["history_encoder"]["max_history_len"]  # 最大历史长度

        self.text_emb_linear = nn.Linear(self.text_emb_size, d_cont_enc)
        self.visual_emb_linear = nn.Linear(self.visual_emb_size, d_cont_enc)
        self.audio_emb_linear = nn.Linear(self.audio_emb_size, d_cont_enc)
        self.speaker_linear = nn.Linear(d_model, d_cont_enc)
        n_speaker = 2
        self.speaker_embedding = nn.Embedding(
            n_speaker,
            model_config.hidden_size,
        )

        self.text_gru = nn.GRU(
            input_size=d_cont_enc,
            hidden_size=d_cont_enc,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # GRU的输出是2*hidden_size，所以需要线性层来将长度映射至hidden_size
        self.text_gru_linear = nn.Sequential(
            nn.Linear(2 * d_cont_enc, d_cont_enc),
            nn.ReLU()
        )
        self.visual_gru = nn.GRU(
            input_size=d_cont_enc,
            hidden_size=d_cont_enc,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # GRU的输出是2*hidden_size，所以需要线性层来将长度映射至hidden_size
        self.visual_gru_linear = nn.Sequential(
            nn.Linear(2 * d_cont_enc, d_cont_enc),
            nn.ReLU()
        )
        self.audio_gru = nn.GRU(
            input_size=d_cont_enc,
            hidden_size=d_cont_enc,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # GRU的输出是2*hidden_size，所以需要线性层来将长度映射至hidden_size
        self.audio_gru_linear = nn.Sequential(
            nn.Linear(2 * d_cont_enc, d_cont_enc),
            nn.ReLU()
        )

        self.context_linear = nn.Linear(d_cont_enc, d_model)
        self.context_attention = SLA(d_model)

    def forward(self, text_emb, visual_emb, audio_emb, speaker,
                history_text_emb, history_visual_emb, history_audio_emb, history_speaker, modal='val'):

        # Embedding
        # 把当前句的文本embedding和对话历史embedding拼接起来
        if 'l' in modal:
            history_text_emb = torch.cat([history_text_emb, text_emb], dim=1)
            history_text_emb = self.text_emb_linear(history_text_emb)
        if 'v' in modal:
            history_visual_emb = torch.cat([history_visual_emb, visual_emb], dim=1)
            history_visual_emb = self.visual_emb_linear(history_visual_emb)
        if 'a' in modal:
            history_audio_emb = torch.cat([history_audio_emb, audio_emb], dim=1)
            history_audio_emb = self.audio_emb_linear(history_audio_emb)

        # # 拼接当前说话人和历史说话人
        history_speaker = torch.cat([history_speaker, speaker], dim=1)
        # # 降维
        history_speaker = self.speaker_linear(self.speaker_embedding(history_speaker))

        # # 将对话文本历史和说话人历史拼接在一起，并进行编码, the reason we dropping this part is same as the above
        if 'l' in modal:
            history_text_enc = torch.cat([history_text_emb, history_speaker], dim=1)
            history_text_con = self.text_gru_linear(self.text_gru(history_text_enc)[0][:, -1, :])
        if 'v' in modal:
            history_visual_enc = torch.cat([history_visual_emb, history_speaker], dim=1)
            history_visual_con = self.visual_gru_linear(self.visual_gru(history_visual_enc)[0][:, -1, :])
        if 'a' in modal:
            history_audio_emb = torch.cat([history_audio_emb, history_speaker], dim=1)
            history_audio_con = self.audio_gru_linear(self.audio_gru(history_audio_emb)[0][:, -1, :])

        if modal == 'val':
            context_enc = torch.stack([history_visual_con, history_audio_con, history_text_con], dim=0)
        elif modal == 'va':
            context_enc = torch.stack([history_visual_con, history_audio_con], dim=0)
        elif modal == 'vl':
            context_enc = torch.stack([history_visual_con, history_text_con], dim=0)
        elif modal == 'al':
            context_enc = torch.stack([history_audio_con, history_text_con], dim=0)
        elif modal == 'v':
            context_enc = history_visual_con.unsqueeze(0)
        elif modal == 'a':
            context_enc = history_audio_con.unsqueeze(0)
        elif modal == 'l':
            context_enc = history_text_con.unsqueeze(0)
        else:
            context_enc = None

        return context_enc


class SLA(nn.Module):
    """ Sequence Level Attention """

    def __init__(self, d_enc):
        super(SLA, self).__init__()
        self.linear = nn.Linear(d_enc, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoding, mask=None):
        attn = self.linear(encoding)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1), -np.inf)
            aux_mask = (attn == -np.inf).all(self.softmax.dim).unsqueeze(self.softmax.dim)
            attn = attn.masked_fill(aux_mask, 0)  # Remove all -inf along softmax.dim
        score = self.softmax(attn).transpose(-2, -1)  # [B, 1, T]
        fused_rep = torch.matmul(score, encoding).squeeze(1)  # [B, d]

        return fused_rep


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
