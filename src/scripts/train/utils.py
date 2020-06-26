import config
import os
import time
import logging
import numpy as np
import h5py
import pandas as pd
import fasttext
import config

def get_word_embedding():
    if config.dataset== 'Atma' or config.dataset == 'ToysFromTrash':
        print('loading word features ...')
        model_embedding = fasttext.load_model(config.word_embedding_path)
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        # call load_data with allow_pickle implicitly set to true
        
        wordtoix = np.load(config.wordtoix_path).tolist()
        ixtoword = np.load(config.ixtoword_path).tolist()
        # restore np.load for future normal usage
        np.load = np_load_old
        word_num = len(wordtoix)
        extract_word_fts = np.random.uniform(-3,3,[word_num,300]) 
        count = 0
        nc = 0
        for index in range(word_num):
            if ixtoword[index] in model_embedding:
                extract_word_fts[index] = model_embedding.get_word_vector(ixtoword[index])
                count = count + 1
                #print(ixtoword[index])
            else:
                extract_word_fts[index] = model_embedding.get_word_vector(ixtoword[index])  # even if the word is not there, fasttext can handle the oov using ngram
                nc = nc + 1

                #print(type(ixtoword[index].decode('utf-8')))
                #print('{}    TCount: {} and Ncount: {}'.format(ixtoword[index],nc+count, nc))
        print('{} words not present in wordembedding out of Total words: {}'.format(nc,count+nc))
        np.save(config.word_fts_path,extract_word_fts)
    else:
        print('loading word features ...')
        word_fts_dict = np.load(open(config.word_embedding_path)).tolist()
        wordtoix = np.load(open(config.wordtoix_path)).tolist()
        ixtoword = np.load(open(config.ixtoword_path)).tolist()
        word_num = len(wordtoix)
        extract_word_fts = np.random.uniform(-3,3,[word_num,300]) 
        count = 0
        for index in range(word_num):
            if ixtoword[index] in word_fts_dict:
                extract_word_fts[index] = word_fts_dict[ ixtoword[index] ]
                count = count + 1
        np.save(config.word_fts_path,extract_word_fts)


def preProBuildWordVocab(logging,sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    logging.info('preprocessing word counts and creating vocab based on word count threshold {:d}'.format(word_count_threshold))
    word_counts = {} # count the word number
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1  # if w is not in word_counts, will insert {w:0} into the dict

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    logging.info('filtered words from {:d} to {:d}'.format(len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


def get_video_data_HL(video_data_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.array(List)


def get_video_data_jukin():
    title = []
    video_list_train = get_video_data_HL(config.video_data_path_train) # get h5 file list
    video_list_val = get_video_data_HL(config.video_data_path_val)
    for ele in video_list_train:
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in range(len(batch_fname)):
            title.append(batch_title[i])
    for ele in video_list_val:
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in range(len(batch_fname)):
            title.append(batch_title[i])
    title = np.array(title)
    video_caption_data = pd.DataFrame({'Description':title})
    return video_caption_data, video_list_train, video_list_val


def make_prepare_path():

    sub_dir = 'alpha_attention'+str(config.alpha_attention)+'_'+'alpha_regress'+str(config.alpha_regress)+'_'+'dimHidden'+str(config.dim_hidden)+'_'+'lr'+str(config.learning_rate)+'_seed'+str(config.random_seed)
    sub_dir = 'ABLR_'+config.dataset+'_'+config.model_mode+'_regressLayer'+ str(config.regress_layer_num) + '_' +sub_dir

    #########################  for logging  #########################################################

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    log_file_name = config.log_dir+'screen_output_'+str(time_stamp)+'_'+sub_dir+'.log'
    fh = logging.FileHandler(filename=log_file_name, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    logging.root.addHandler(fh)

    model_save_path = config.model_save_dir+sub_dir+"/"+time_stamp
    result_save_path = config.result_save_dir+sub_dir+"/"+time_stamp

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    if not os.path.exists(config.words_dir):
        os.makedirs(config.words_dir)

    return sub_dir, logging, config.regress_layer_num, model_save_path, result_save_path
