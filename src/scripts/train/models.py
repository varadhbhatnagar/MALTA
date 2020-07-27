import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import pdb

class Attention(nn.Module):
    def __init__(self, dim_hidden_attendee, dim_hidden_attendant):
        super(Attention, self).__init__()
        self.embed_att_w = torch.zeros(dim_hidden_attendee,1).cuda()
        torch.nn.init.xavier_uniform_(self.embed_att_w)
        self.embed_att_Wa = torch.zeros(dim_hidden_attendant, dim_hidden_attendee).cuda()
        torch.nn.init.xavier_uniform_(self.embed_att_Wa)
        self.embed_att_Ua = torch.zeros(dim_hidden_attendee, dim_hidden_attendee).cuda()
        torch.nn.init.xavier_uniform_(self.embed_att_Ua)
        self.embed_att_ba = torch.zeros(dim_hidden_attendee).cuda()

    def forward(self, attendee_fts, dim_hidden_attendee, attendee_step_size, attendant_fts, video_mask):
        brcst_w = self.embed_att_w.unsqueeze(0).repeat(attendee_step_size,1,1) # n x h x 1
        attendee_part = torch.matmul(attendee_fts.float(), self.embed_att_Ua.unsqueeze(0).repeat(attendee_step_size,1,1)) + self.embed_att_ba # n x b x h
        e = torch.tanh(torch.matmul(attendant_fts.float(), self.embed_att_Wa) + attendee_part)
        e = torch.matmul(e, brcst_w)
        e = torch.sum(e,dim = 2) # n x b
        e_hat_exp = torch.mul(video_mask.T, torch.exp(e)) # n x b
        denomin = torch.sum(e_hat_exp,dim = 0) # b
        denomin = denomin + denomin.eq(0).float()   # regularize denominator
        alphas = torch.div(e_hat_exp,denomin).unsqueeze(2).repeat(1,1,dim_hidden_attendee) # n x b x h  # normalize to obtain alpha
        attention_list = alphas* attendee_fts # n x b x h ; multiply is element x element
        attended_fts = torch.sum(attention_list, dim = 0) # b x h       #  visual feature by soft-attention weighted sum
        predict_attention_weights = torch.transpose(torch.div(e_hat_exp,denomin),0,1)  # b x n

        return attended_fts, predict_attention_weights
	
	
class ConcAV(nn.Module):
    
    def __init__(self, weights_matrix):
        super(ConcAV, self).__init__()

        # Caption Embedding Layer
        self.caption_embedding = nn.Embedding(weights_matrix.shape[0], embedding_dim = config.dim_word, padding_idx = 0)
        self.caption_embedding_dropout = nn.Dropout(config.rnn_dropout)
        self.caption_embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.caption_embedding.requires_grad = False

        # LSTMs     
        self.video_rnn = nn.LSTM(input_size= config.dim_image, hidden_size=config.dim_hidden_video , num_layers= config.rnn_nlayers, dropout=config.rnn_dropout, batch_first=True, bidirectional=True)
        self.audio_rnn = nn.LSTM(input_size= config.dim_audio , hidden_size=config.dim_hidden_audio , num_layers=config.rnn_nlayers, dropout=config.rnn_dropout, batch_first=True, bidirectional=True)
        self.caption_rnn = nn.LSTM(input_size= config.dim_word, hidden_size=config.dim_hidden , num_layers=config.rnn_nlayers , dropout=config.rnn_dropout, batch_first=True, bidirectional=True)
        
        # # LSTM Output linear transform layers
        self.video_linear_transform = nn.Linear(2*config.dim_hidden_video, config.dim_hidden_video)
        self.caption_linear_transform = nn.Linear(2*config.dim_hidden, config.dim_hidden)
        self.audio_linear_transform = nn.Linear(2*config.dim_hidden_audio, config.dim_hidden_audio)

        # Attention Layers
        self.video_using_sentence_attention = Attention(config.dim_hidden_video, config.dim_hidden)
        self.audio_using_sentence_attention = Attention(config.dim_hidden_audio, config.dim_hidden)
        self.sentence_using_audio_video_attention = Attention(config.dim_hidden, config.dim_hidden_audio + config.dim_hidden_video)
        self.video_using_attendend_sentence_attention = Attention(config.dim_hidden_video, config.dim_hidden)
        self.audio_using_attendend_sentence_attention = Attention(config.dim_hidden_audio, config.dim_hidden)


        # Prediction Layers
        self.pred_dropout = nn.Dropout()
        self.fc1 = nn.Linear(config.n_frame_step, config.dim_hidden_regress)
        self.fc2 = nn.Linear(config.dim_hidden_regress, 2)


    def forward(self, vid, aud, cap, cap_embedding_mask, vid_mask, aud_mask, cap_mask):
        aud_mask = torch.from_numpy(aud_mask).cuda()
        vid_mask = torch.from_numpy(vid_mask).cuda()
        cap_mask = torch.from_numpy(cap_mask).cuda()
        cap_embedding_mask = torch.from_numpy(cap_embedding_mask).cuda()

        # Video LSTM
        vid = torch.from_numpy(vid).float().cuda()
        vid, _ = self.video_rnn(vid)
        vid_dim = vid.size()
        vid = self.video_linear_transform(vid.contiguous().view(-1, 2*config.dim_hidden_video))
        vid = F.relu(vid)
        vid = vid.view(vid_dim[0], vid_dim[1], int(vid_dim[2]/2))

        # Audio LSTM
        aud = torch.from_numpy(aud).float().cuda()
        aud, _ = self.audio_rnn(aud)
        aud_dim = aud.size()
        aud = self.audio_linear_transform(aud.contiguous().view(-1, 2*config.dim_hidden_audio))
        aud = F.relu(aud)
        aud = aud.view(aud_dim[0], aud_dim[1], int(aud_dim[2]/2))

        # Caption LSTM
        cap = torch.from_numpy(cap).long().cuda()
        cap = self.caption_embedding(cap)
        cap = self.caption_embedding_dropout(cap)
        cap, _ = self.caption_rnn(cap)
        cap_dim = cap.size()
        cap = self.caption_linear_transform(cap.contiguous().view(-1, 2*config.dim_hidden))
        cap = F.relu(cap)
        cap = cap.view(cap_dim[0], cap_dim[1], int(cap_dim[2]/2))
        cap = cap_embedding_mask*cap.permute(1,0,2)
        cap = cap.permute(1,2,0)
        cap_mean = cap.mean(dim=2)

        vid_temp = vid.permute(1,0,2)
        aud_temp = aud.permute(1,0,2)

        # Alternating Attention 
        attended_video_fts, predict_attention_weights_v  = self.video_using_sentence_attention.forward(vid_temp, config.dim_hidden_video,  config.n_frame_step, cap_mean, vid_mask)
        attended_audio_fts, predict_attention_weights_a = self.audio_using_sentence_attention.forward(aud_temp, config.dim_hidden_audio,  config.n_frame_step, cap_mean, aud_mask)
        attended_sentence_fts, predict_attention_weights_q_a = self.sentence_using_audio_video_attention.forward(cap_mean, config.dim_hidden,  config.n_caption_step, torch.cat((attended_video_fts, attended_audio_fts), dim=1), cap_mask)
        predict_attention_weights_q_v = predict_attention_weights_q_a
        attended_video_fts, predict_attention_weights_v = self.video_using_attendend_sentence_attention.forward(vid_temp, config.dim_hidden_video, config.n_frame_step, attended_sentence_fts, vid_mask)
        attended_audio_fts, predict_attention_weights_a = self.audio_using_attendend_sentence_attention.forward(aud_temp, config.dim_hidden_audio, config.n_frame_step, attended_sentence_fts, aud_mask)


        # Regression Module
        # TODO : Incorporate AF model too
        # TODO : Dropout in FC layers, single layer model

        multimodal_fts_concat = torch.add(predict_attention_weights_v, predict_attention_weights_a).float().cuda()
        # if config.regress_layer_num ==1:
        #     predict_location = self.fc1(multimodal_fts_concat)
        #     predict_location = F.relu(predict_location)
        # else:
        predict_hidden = self.fc1(multimodal_fts_concat)
        predict_location = F.relu(self.fc2(predict_hidden))

        return predict_location, predict_attention_weights_v, predict_attention_weights_a


