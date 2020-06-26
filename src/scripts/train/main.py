import torch
import torch.nn.functional as F
from models import ConcAV
import config
from utils import make_prepare_path, get_video_data_jukin, preProBuildWordVocab, get_word_embedding
import numpy as np
import string
import pandas as pd
import time
import h5py
from keras.preprocessing import sequence
import pdb

def my_loss(predict_location, video_location, predict_attention_weights_v, video_attention_weights, predict_attention_weights_a, audio_attention_weights, loss_mask):
    loss_mask = torch.from_numpy(loss_mask).float()
    loss_regression_temp = F.smooth_l1_loss(predict_location, torch.from_numpy(video_location).float())
    loss_regression = config.alpha_regress * loss_regression_temp.mean()

    batch_ground_weight = torch.from_numpy(video_attention_weights).float() * torch.log(predict_attention_weights_v+1e-12)
    batch_video_attention_weight = torch.sum(batch_ground_weight,dim = 1)
    loss_attention_v = -torch.mean( loss_mask * batch_video_attention_weight)
    loss_attention_v = config.alpha_attention *loss_attention_v

    batch_ground_weight = torch.from_numpy(audio_attention_weights).float() * torch.log(predict_attention_weights_a+1e-12)
    batch_audio_attention_weight = torch.sum(batch_ground_weight,dim = 1)
    loss_attention_a = -torch.mean( loss_mask * batch_audio_attention_weight)
    loss_attention_a = config.alpha_attention *loss_attention_a

    return loss_regression + loss_attention_v + loss_attention_a, loss_regression, loss_attention_v


def train(sub_dir, logging, train_data, wordtoix, ixtoword, word_emb_init):

    model = ConcAV(word_emb_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    all_batches_count = 0
    tStart_total = time.time()

    for epoch in range(config.n_epochs):
        
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        tStart_epoch = time.time()
        loss_epoch = np.zeros(len(train_data)) # each item in loss_epoch record the loss of this h5 file

        for current_batch_file_idx in range(len(train_data)):
            
            all_batches_count = all_batches_count +  1
            logging.info("current_batch_file_idx = {:d}".format(current_batch_file_idx))
            tStart = time.time()
            current_batch = h5py.File(train_data[current_batch_file_idx],'r')
            current_captions_tmp = current_batch['title']
            current_captions = []
            for ind in range(config.batch_size):
                current_captions.append(current_captions_tmp[ind].decode())
            current_captions = np.array(current_captions)
            for ind in range(config.batch_size):
                for c in string.punctuation: 
                    current_captions[ind] = current_captions[ind].replace(c,'')
            current_feats = np.zeros((config.batch_size, config.n_frame_step, config.dim_image))
            current_feats_audio = np.zeros((config.batch_size, config.n_frame_step, config.dim_audio))
            current_weights = np.zeros((config.batch_size,config.n_frame_step))
            current_video_masks = np.zeros((config.batch_size, config.n_frame_step))
            current_video_location = np.zeros((config.batch_size,2))
            current_loss_mask = np.ones(config.batch_size)+0.0
            for i in range(config.batch_size):
                current_captions[i] = current_captions[i].strip()
                if current_captions[i] == '':
                    current_captions[i] = '.'
                    current_loss_mask[i] = 0.0
            for ind in range(config.batch_size):
                current_feats[ind,:,:] = current_batch['data'][:config.n_frame_step,ind,:4096]
                current_feats_audio[ind,:,:] = current_batch['data'][:config.n_frame_step,ind,4096:]
                
                current_video_location[ind,:] = np.array(current_batch['norm_timestamps'][ind])
                if config.dataset == 'ActivityNet':
                    current_weights[ind,:] = np.array(current_batch['weights'][:config.n_frame_step,ind])
                elif config.dataset == 'charades':
                    current_weights[ind,:] = np.array(current_batch['weights'][:config.n_frame_step,ind])
                elif config.dataset == 'ToysFromTrash':
                    current_weights[ind,:] = np.array(current_batch['weights'][:config.n_frame_step,ind])
                elif config.dataset == 'Atma':
                    current_weights[ind,:] = np.array(current_batch['weights'][:config.n_frame_step,ind])
                elif config.dataset == 'TACOS':
                    left = int(current_video_location[ind][0]*config.n_frame_step)
                    right = min(config.n_frame_step,int(current_video_location[ind][1]*config.n_frame_step)+1)
                    current_weights[ind,left:right] = 1.0 / (right - left + 1)
                idx = np.where(current_batch['label'][:,ind] != -1)[0] #find this video's segment finish point
                if len(idx) == 0:
                    continue
                current_video_masks[ind,:idx[-1]+1] = 1

            current_audio_masks = current_video_masks
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)
            # TODO : Replace Keras Preprocessing Function by something appropriate in Pytorch
            current_caption_matrix = sequence.pad_sequences(list(current_caption_ind), padding='post', maxlen=config.n_caption_step-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( list(map(lambda x: (x != 0).sum(), current_caption_matrix ))) # save the sentence length of this batch

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1
            current_caption_emb_mask = np.zeros((config.n_caption_step,config.batch_size,config.dim_hidden)) + 0.0
            for ii in range(config.batch_size):
                current_caption_emb_mask[:nonzeros[ii],ii,:] = 1.0 / (nonzeros[ii]*1.0)

            reverse_current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[::-1] if word in wordtoix], current_captions)
            # TODO: Same as above todo
            reverse_current_caption_matrix = sequence.pad_sequences(list(reverse_current_caption_ind), padding='post', maxlen=config.n_caption_step-1)
            reverse_current_caption_matrix = np.hstack( [reverse_current_caption_matrix, np.zeros( [len(reverse_current_caption_matrix),1]) ] ).astype(int)

            current_caption_length = np.zeros((config.batch_size))
            for ind in range(config.batch_size):
                current_caption_length[ind] = np.sum(current_caption_masks[ind])

            predict_location, predict_attention_weights_v, predict_attention_weights_a = model.forward(vid = current_feats, aud = current_feats_audio, cap = current_caption_matrix, cap_embedding_mask = current_caption_emb_mask, vid_mask = current_video_masks, aud_mask = current_audio_masks, cap_mask = current_caption_masks)
            loss_val, loss_regression, loss_attention_v = my_loss(predict_location, current_video_location, predict_attention_weights_v, current_weights,  predict_attention_weights_a, current_weights, current_loss_mask)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_epoch[current_batch_file_idx] = loss_val

            logging.info("loss = {:f}  loss_regression = {:f}  loss_attention = {:f}".format(loss_val,loss_regression,loss_attention_v))
            tStop = time.time()

        logging.info("Epoch: {:d} done.".format(epoch))
        tStop_epoch = time.time()
        logging.info('Epoch Time Cost: {:f} s'.format(round(tStop_epoch - tStart_epoch,2)))


def main():
    sub_dir, logging, regress_layer_num, model_save_dir, result_save_dir = make_prepare_path()

    meta_data, train_data, val_data = get_video_data_jukin()
    if config.task == 'train':
        captions = meta_data['Description'].values
        for i in range(len(captions)):
            captions[i] = captions[i].decode()
        for c in string.punctuation:
            captions = map(lambda x: x.replace(c, ''), captions)

        wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(logging, captions, word_count_threshold=1)

        np.save(config.ixtoword_path, ixtoword)
        np.save(config.wordtoix_path, wordtoix)

        get_word_embedding()
        word_emb_init = np.array(np.load(config.word_fts_path).tolist(),np.float32)
        print(word_emb_init.shape)
        logging.info('regress_layer_num = {:f}'.format(regress_layer_num*1.0))

        train(sub_dir, logging, train_data, wordtoix, ixtoword, word_emb_init)
        #df = pd.read_csv("max_values_coattention_"+str(config.random_seed)+".csv")
        #df.to_csv('result.csv', mode='a+', header=False)
    
    # elif config.task == 'localize':
    #     word_emb_init = np.array(np.load(open(config.word_fts_path)).tolist(),np.float32)

    #     localize(sub_dir, logging, regress_layer_num, result_save_dir = result_save_dir, model_path = config.model)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(config.random_seed)
    gpu_id=config.gpu_id
    main()
