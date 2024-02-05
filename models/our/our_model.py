import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier, Fusion
from models.networks.multihead_attention import MultiheadAttention
from models.networks.ContextEncoder import ConversationalContextEncoder
from models.networks.interact_model import InteractModule
from models.utils.config import OptConfig
from models.networks.tools import get_mask_from_lengths
import math
import time
import numpy as np
from models.pretrain_model import pretrainModel


class ourModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='visual embedding method,last,mean or atten')
        # parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--data_path', type=str,
                            help='where to load dataset')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--focal_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--temperature', type=float, default=0.007, help='temperature of contrastive learning loss')
        # parameter of pretrain
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--best_cvNo', type=int, default=1, help='best cvNo of pretrain model')
        # parameter of Transformer Encoder
        parser.add_argument('--Transformer_head', type=int, default=2, help='head of Transformer_head')
        parser.add_argument('--Transformer_layers', type=int, default=1, help='layer of Transformer_head')
        # parameter of multi-head attention
        parser.add_argument('--attention_head', type=int, default=1, help='head of multi-head attention')
        parser.add_argument('--attention_dropout', type=float, default=0., help='head of multi-head attention')
        # parameter of ContextEncoder
        parser.add_argument('--ContextEncoder_layers', type=int, default=2,
                            help='the layers of ContextEncoder')
        parser.add_argument('--ContextEncoder_dropout', type=float, default=0.2,
                            help='the layers of ContextEncoder')
        parser.add_argument('--ContextEncoder_max_history_len', type=int, default=10,
                            help='the layers of ContextEncoder')
        # other parameter
        parser.add_argument('--activate_fun', type=str, default='relu', help='which activate function will be used')
        parser.add_argument('--ablation', type=str, default='normal', help='which module should be ablate')
        parser.add_argument('--use_ICL', type=bool, help='add imbalance classify loss or not')
        parser.add_argument('--drop_last', type=bool, default=False, help='drop the last data or not')

        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = []
        self.model_names = []  # 所有模块的名称

        # acoustic model
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.model_names.append('EmoA')
        self.netIntA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.model_names.append('IntA')

        # lexical model
        self.netEmoL = TextCNN(opt.input_dim_l, opt.embd_size_l, dropout=0.5)
        self.model_names.append('EmoL')
        self.netIntL = TextCNN(opt.input_dim_l, opt.embd_size_l, dropout=0.5)
        self.model_names.append('IntL')

        # visual model
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('EmoV')
        self.netIntV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('IntV')

        # Transformer Fusion model
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head))
        self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')
        int_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head))
        self.netIntFusion = torch.nn.TransformerEncoder(int_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('IntFusion')

        # Modality Interaction
        self.netEmo_Int_interaction = InteractModule(opt)
        self.model_names.append('Emo_Int_interaction')
        self.netInt_Emo_interaction = InteractModule(opt)
        self.model_names.append('Int_Emo_interaction')

        # Classifier
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = 3 * opt.hidden_size
        # 考虑对话历史的话，应该就不需要BN层了
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netIntC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.int_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('IntC')
        self.loss_names.append('int_CE')

        # Context
        self.netContext = ConversationalContextEncoder(model_config=opt)
        self.model_names.append('Context')

        self.temperature = opt.temperature

        if self.isTrain:
            # self.load_pretrained_encoder(opt)
            if not opt.use_ICL:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = torch.nn.CrossEntropyLoss()
            else:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight
            # self.cl_weight = opt.cl_weight
        # else:
        #     self.load_pretrained_encoder(opt)

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)


    # 加载预训练Encoder，
    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format(opt.pretrained_path))
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.best_cvNo))
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')
        pretrained_config = self.load_from_opt_record(pretrained_config_path)
        pretrained_config.isTrain = False                             # teacher model should be in test mode
        pretrained_config.gpu_ids = opt.gpu_ids                       # set gpu to the same
        self.pretrained_encoder = pretrainModel(pretrained_config)
        self.pretrained_encoder.load_networks_cv(pretrained_path)
        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()

    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.'+key, value) for key, value in state_dict.items()])
        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoL.load_state_dict(f(self.pretrained_encoder.netEmoL.state_dict()))
            self.netIntA.load_state_dict(f(self.pretrained_encoder.netIntA.state_dict()))
            self.netIntV.load_state_dict(f(self.pretrained_encoder.netIntV.state_dict()))
            self.netIntL.load_state_dict(f(self.pretrained_encoder.netIntL.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))
            self.netIntFusion.load_state_dict(f(self.pretrained_encoder.netIntFusion.state_dict()))


    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.lexical = input['L_feat'].float().to(self.device)

        # Emotion label
        self.emo_label = input['emo_label'].to(self.device)
        # Intent label
        self.int_label = input['int_label'].to(self.device)

        if self.opt.use_history:
            self.speaker = input['speakers']
            self.history_text_embs = input['history_text_embs']
            self.history_visual_embs = input['history_visual_embs']
            self.history_audio_embs = input['history_audio_embs']
            self.history_speaker = input['history_speaker_embs']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # 处理当前信息
        emo_feat_A = self.netEmoA(self.acoustic)
        emo_feat_L = self.netEmoL(self.lexical)
        emo_feat_V = self.netEmoV(self.visual)

        int_feat_A = self.netIntA(self.acoustic)
        int_feat_L = self.netIntL(self.lexical)
        int_feat_V = self.netIntV(self.visual)

        emo_fusion_feat = torch.stack((emo_feat_V, emo_feat_A, emo_feat_L), dim=0)
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)

        int_fusion_feat = torch.stack((int_feat_V, int_feat_A, int_feat_L), dim=0)
        int_fusion_feat = self.netIntFusion(int_fusion_feat)

        if self.opt.use_history:
            history_info = self.netContext(
                self.lexical, self.visual, self.acoustic, self.speaker,
                self.history_text_embs, self.history_visual_embs, self.history_audio_embs, self.history_speaker
            )
            emo_fusion_feat += history_info
            int_fusion_feat += history_info

        emo_final_feat = self.netEmo_Int_interaction(emo_fusion_feat, int_fusion_feat, int_fusion_feat)
        int_final_feat = self.netInt_Emo_interaction(int_fusion_feat, emo_fusion_feat, emo_fusion_feat)

        # emotion prediction
        self.emo_logits, _ = self.netEmoC(emo_final_feat)

        # intent prediction
        self.int_logits, _ = self.netIntC(int_final_feat)

        self.emo_pred = F.softmax(self.emo_logits, dim=-1)
        self.int_pred = F.softmax(self.int_logits, dim=-1)

        # print(f'forward need {time.time() - forward_time}s')

    def backward(self):
        """Calculate the loss for back propagation"""
        # backward_time = time.time()
        self.loss_emo_CE = self.ce_weight * self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_int_CE = self.ce_weight * self.criterion_ce(self.int_logits, self.int_label)
        #
        loss = self.loss_emo_CE + self.loss_int_CE # + self.loss_EmoF_CE + self.loss_IntF_CE

        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

        # print(f'calculate loss and backward need {time.time() - backward_time}s')

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # print('forward has done')
        # backward
        self.optimizer.zero_grad()
        self.backward()
        # print('backward has done')
        self.optimizer.step()


class ActivateFun(torch.nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)
