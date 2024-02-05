import os
import json
import random
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset
import time


class MultimodalHistoryDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--emo_output_dim', type=int, default=7, help='how many label types in this dataset')
        parser.add_argument('--int_output_dim', type=int, default=9, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'],
                            help='how to normalize input comparE feature')
        parser.add_argument('--corpus_name', type=str, default='IEMOCAP', help='which dataset to use')
        parser.add_argument('--use_history', type=bool, help='add history information or not')
        return parser

    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', f'{opt.corpus_name}_config.json')))
        self.norm_method = opt.norm_method
        self.corpus_name = opt.corpus_name
        self.opt = opt

        # load feature, 这里的数据加载方式可能得换一下
        self.A_type = opt.A_type
        self.all_A = h5py.File(os.path.join(config['feature_root'], 'A', f'{self.A_type}.h5'), 'r')

        self.V_type = opt.V_type
        self.all_V = h5py.File(os.path.join(config['feature_root'], 'V', f'{self.V_type}.h5'), 'r')

        self.L_type = opt.L_type
        self.all_L = h5py.File(os.path.join(config['feature_root'], 'L', f'{self.L_type}.h5'), 'r')

        self.Dia_Index = self.h5_to_dict(h5py.File(os.path.join(config['feature_root'], "Dia_Index", 'dia_index.h5'), 'r'))

        self.all_speaker = self.h5_to_dict(
            h5py.File(os.path.join(config['feature_root'], 'Speaker', 'speaker.h5'))
        )

        self.all_L_keys = self.all_L.keys()

        # load target
        emotion_label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_emotion_label.npy")
        intent_label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_intent_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.emotion_label = np.load(emotion_label_path)
        self.intent_label = np.load(intent_label_path)

        # print(self.emotion_label, self.emotion_label.shape)
        self.emotion_label = np.argmax(self.emotion_label, axis=1)
        self.intent_label = np.argmax(self.intent_label, axis=1)
        self.int2name = np.load(int2name_path)

        # set collate function
        self.manual_collate_fn = True

        # set conversation history
        self.history_type = opt.use_history  # every string is ok but not 'none'
        self.max_history_len = opt.ContextEncoder_max_history_len

    def h5_to_dict(self, h5f):
        ret = {}
        for key in h5f.keys():
            ret[key] = h5f[key][()]
        return ret

    def __getitem__(self, index):
        int2name = self.int2name[index]
        if self.corpus_name == 'IEMOCAP':
            int2name = int2name[0].decode()
        emo_label = torch.tensor(self.emotion_label[index])
        int_label = torch.tensor(self.intent_label[index])

        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        # process V_feat
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # proveee L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
        speaker = self.all_speaker[int2name]

        # History
        basename = int2name.split('_')
        dialog = f'{basename[0]}_{basename[1]}'  # 切分对话序号
        turn = int(basename[-1])  # 切分对话内语句序号
        history_len = min(self.max_history_len, turn)  # 当前对话历史长度
        history_text_emb = list()  # 历史embedding，初始为空列表
        history_audio_emb = list()
        history_visual_emb = list()
        history_speaker = list()

        # 1、这里只需要提取文本的对话历史即可
        # 2、所有的文本都存储在h5文件中
        # 3、怎样通过对话编号，找到h5文件中对应对话的所有语句
        if self.history_type:
            # 提取同一对话场景下所有语句的文件名，并按照语句顺序排列
            # 这里可以优化一下
            # history_basenames = []
            # for key in self.all_L_keys:
            #     if dialog in key:
            #         history_basenames.append(key)
            history_basenames = self.Dia_Index[dialog]

            # 截取最近一段对话历史的文件名
            history_basenames = history_basenames[:turn][-history_len:]

            for i, h_basename in enumerate(history_basenames):  # 提取并记录每个历史对话的说话人信息、文本embedding
                h_basename = h_basename.decode('utf-8')
                h_text_emb = torch.from_numpy(self.all_L[h_basename][()]).float()
                h_audio_emb = torch.from_numpy(self.all_A[h_basename][()]).float()
                h_visual_emb = torch.from_numpy(self.all_V[h_basename][()]).float()
                h_speaker = self.all_speaker[h_basename]
                # 这里要加一段speaker的导入

                # print(h_text_emb.shape)

                history_text_emb.append(h_text_emb)
                history_audio_emb.append(h_audio_emb)
                history_visual_emb.append(h_visual_emb)

                history_speaker.append(h_speaker)
                # 这里要将导入的speaker添加到list中

                # Padding
                if i == history_len - 1 and history_len < self.max_history_len:
                    self.pad_history(
                        self.max_history_len - history_len,
                        history_speaker=history_speaker,
                        history_text_emb=history_text_emb,
                        history_audio_emb=history_audio_emb,
                        history_visual_emb=history_visual_emb
                    )
            if turn == 0:  # 如果当前句为对话场景第一句，则按照最大对话长度进行历史padding
                self.pad_history(
                    self.max_history_len,
                    history_speaker=history_speaker,
                    history_text_emb=history_text_emb,
                    history_audio_emb=history_audio_emb,
                    history_visual_emb=history_visual_emb
                )

            history_speaker = np.array(history_speaker)
            # print(f'history_speaker.shape is: {history_speaker.shape}')
            history_text_emb = torch.cat(history_text_emb, dim=0)
            # history_text_emb = pad_sequence(history_text_emb, True, 0)
            # history_text_emb = np.array(history_text_emb)

            history_visual_emb = torch.cat(history_visual_emb)
            # history_visual_emb = pad_sequence(history_visual_emb, True, 0)
            # history_visual_emb = np.array(history_visual_emb)

            history_audio_emb = torch.cat(history_audio_emb)
            # history_audio_emb = pad_sequence(history_audio_emb, True, 0)
            # history_audio_emb = np.array(history_audio_emb)

            history = {  # 对话历史不为空的情况下，应该包括历史长度、历史embedding_list
                "history_len": history_len,
                "history_speakers": history_speaker,
                "history_text_emb": history_text_emb,
                "history_audio_emb": history_audio_emb,
                "history_visual_emb": history_visual_emb,
            }

            return {
                'A_feat': A_feat,
                'V_feat': V_feat,
                'L_feat': L_feat,
                'emo_label': emo_label,
                'int_label': int_label,
                # 'int2name': int2name,
                'history': history,
                'speaker': speaker,
            }
        else:
            return {
                'A_feat': A_feat,
                'V_feat': V_feat,
                'L_feat': L_feat,
                'emo_label': emo_label,
                'int_label': int_label,
            }


    def pad_history(self,
                    pad_size,
                    history_text_emb=None,
                    history_visual_emb=None,
                    history_audio_emb=None,
                    history_text_len=None,
                    history_emotion=None,
                    history_speaker=None,
                    ):
        for _ in range(pad_size):  # 如果参数列表里的各项不是None，则填补1维0向量
            history_text_emb.insert(0, torch.from_numpy(
                np.zeros((1, self.opt.input_dim_l),
                         dtype=np.float32))) if history_text_emb is not None else None
            history_visual_emb.insert(0, torch.from_numpy(
                np.zeros((1, self.opt.input_dim_v),
                         dtype=np.float32))) if history_visual_emb is not None else None
            history_audio_emb.insert(0, torch.from_numpy(
                np.zeros((1, self.opt.input_dim_a),
                         dtype=np.float32))) if history_audio_emb is not None else None
            history_speaker.insert(0,
                                   0) if history_speaker is not None else None  # meaningless zero padding, should be cut out by mask of history_len

            history_text_len.insert(
                0) if history_text_len is not None else None  # meaningless zero padding, should be cut out by mask of history_len
            history_emotion.append(
                0) if history_emotion is not None else None  # meaningless zero padding, should be cut out by mask of history_len

    def __len__(self):
        # assert len(self.emotion_label) == len(self.intent_label)
        # return len(self.emotion_label)
        return len(self.intent_label)

    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]

        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)

        emo_label = torch.tensor([sample['emo_label'] for sample in batch]).long()
        int_label = torch.tensor([sample['int_label'] for sample in batch]).long()

        if self.history_type:
            speakers = torch.tensor([[sample['speaker']] for sample in batch])
            # L_len = np.array([text.shape[0] for text in L])
            # V_len = np.array([v.shape[0] for v in V])
            # A_len = np.array([a.shape[0] for a in A])

            # lengths = torch.tensor([len(sample) for sample in A]).long()
            # int2name = [sample['int2name'] for sample in batch]

            history_lens = [sample["history"]["history_len"] for sample in batch]
            # history_text_emb中存储的是每个对话的对话历史，不同对话场景的对话数量不同，则对话历史长度也不相同
            # 太长的对话历史会被裁剪为max_history_len，而不足的则保留原始长度
            # history_text_embs是list类型数据，用来存储所有对话场景的对话历史。因为每个对话场景的历史长度不同，所以history_text_embs是一个不等长的二维list
            # 所以history_text_embs在转为np.array时，会报警告
            history_text_embs = [sample["history"]["history_text_emb"] for sample in batch]
            history_visual_embs = [sample["history"]["history_visual_emb"] for sample in batch]
            history_audio_embs = [sample["history"]["history_audio_emb"] for sample in batch]
            # 这里要加说话人的历史信息
            history_speaker_embs = [sample['history']["history_speakers"] for sample in batch]

            history_text_embs = pad_sequence(history_text_embs, batch_first=True, padding_value=0)
            history_audio_embs = pad_sequence(history_audio_embs, batch_first=True, padding_value=0)
            history_visual_embs = pad_sequence(history_visual_embs, batch_first=True, padding_value=0)

            history_lens = torch.tensor(history_lens)
            try:
                history_speaker_embs = torch.tensor(history_speaker_embs)
            except:
                _history_speaker_embs = []
                for speaker in history_speaker_embs:
                    if self.max_history_len - speaker.shape[0]:
                        for _ in range(self.max_history_len - speaker.shape[0]):
                            speaker = np.insert(speaker, 0, 0)
                    _history_speaker_embs.append(speaker)
                history_speaker_embs = torch.tensor(_history_speaker_embs)

            return {
                'A_feat': A,
                'V_feat': V,
                'L_feat': L,
                'emo_label': emo_label,
                'int_label': int_label,
                'speakers': speakers,
                # 'lengths': lengths,
                # 'int2name': int2name,
                'history_text_embs': history_text_embs,
                'history_visual_embs': history_visual_embs,
                'history_audio_embs': history_audio_embs,
                'history_speaker_embs': history_speaker_embs,
                'history_lens': history_lens
            }
        else:
            return {
                'A_feat': A,
                'V_feat': V,
                'L_feat': L,
                'emo_label': emo_label,
                'int_label': int_label,
            }

