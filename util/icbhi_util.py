from collections import namedtuple
import os
import math
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torchaudio
from torchaudio import transforms as T
from .meta_description import generate_meta_description

__all__ = ['get_annotations', 'get_individual_cycles_torchaudio', 'generate_mel_spectrogram', 'generate_fbank', 'get_score']


# ==========================================================================
""" ICBHI dataset information """
def _extract_lungsound_annotation(file_name, data_folder):
    tokens = file_name.strip().split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient Number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_folder, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')

    return recording_info, recording_annotations


def get_annotations(args, data_folder):
    if args.class_split == 'lungsound' or args.class_split in ['lungsound_meta', 'meta']: # 4-class
        filenames = sorted(glob(data_folder+'/*')) #--> 1840: 920 wav + 920 txt
        filenames = set(f.strip().split('/')[-1].split('.')[0] for f in filenames if '.txt' in f)
        filenames = sorted(list(set(filenames))) #--> 920

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            annotation_dict[f] = ann

    elif args.class_split == 'diagnosis':
        filenames = sorted(glob(data_folder+'/*')) #--> 1840: 920 wav + 920 txt
        filenames = set(f.strip().split('/')[-1].split('.')[0] for f in filenames if '.txt' in f)
        filenames = sorted(list(set(filenames))) #--> 920
        tmp = pd.read_csv(os.path.join(args.data_folder, 'icbhi_dataset/patient_diagnosis.txt'), names=['Disease'], delimiter='\t')

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            ann.drop(['Crackles', 'Wheezes'], axis=1, inplace=True)

            disease = tmp.loc[int(f.strip().split('_')[0]), 'Disease']
            ann['Disease'] = disease

            annotation_dict[f] = ann
            
    return annotation_dict

""" data preprocessing """

def _get_lungsound_label(crackle, wheeze, n_cls, args):
    if n_cls == 4 or args.method in ['mscl']:
        if crackle == 0 and wheeze == 0:
            return 0
        elif crackle == 1 and wheeze == 0:
            return 1
        elif crackle == 0 and wheeze == 1:
            return 2
        elif crackle == 1 and wheeze == 1:
            return 3
    
    elif n_cls == 2:
        if crackle == 0 and wheeze == 0:
            return 0
        else:
            return 1


def _get_diagnosis_label(disease, n_cls):
    if n_cls == 3:
        if disease in ['COPD', 'Bronchiectasis', 'Asthma']:
            return 1
        elif disease in ['URTI', 'LRTI', 'Pneumonia', 'Bronchiolitis']:
            return 2
        else:
            return 0

    elif n_cls == 2:
        if disease == 'Healthy':
            return 0
        else:
            return 1


def _slice_data_torchaudio(start, end, data, sample_rate):
    """
    SCL paper..
    sample_rate denotes how many sample points for one second
    """
    max_ind = data.shape[1]
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[:, start_ind: end_ind]


def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
        if data.dim() == 1:
            data = data.unsqueeze(0)
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data

def get_meta_text_descriptions(metadata, args, training=False):
    age = int(metadata[0])
    sex = int(metadata[1])
    loc = int(metadata[5])    
    dev = int(metadata[6])
    
    
    if age >= 19:
        age_str = 'adult'
        age_tmp = 0
    else:
        age_str = 'pediatric'
        age_tmp = 1
    
    if sex == 0:
        sex_str = 'male'
    else:
        sex_str = 'female'
    #'Tc':0, 'Al':1, 'Ar':2, 'Pl':3, 'Pr':4, 'Ll':5, 'Lr':6
    if loc == 0:
        loc_str = 'trachea'
    elif loc == 1:
        loc_str = 'left anterior chest'
    elif loc == 2:
        loc_str = 'right anterior chest'
    elif loc == 3:
        loc_str = 'left posterior chest'
    elif loc == 4:
        loc_str = 'right posterior chest'
    elif loc == 5:
        loc_str = 'left lateral chest'
    elif loc == 6:
        loc_str = 'right lateral chest'
    
    #Meditron': 0, 'LittC2SE': 1, 'Litt3200': 2, 'AKGC417L': 3
    if dev == 0:
        dev_str = 'Meditron'
    elif dev == 1:
        dev_str = 'LittC2SE'
    elif dev == 2:
        dev_str = 'Litt3200'
    elif dev == 3:
        dev_str = 'AKGC417L'
    
    age_str_list = []
    sex_str_list = []
    loc_str_list = []
    dev_str_list = []
    
    if args.test_wrong_label:        
        age_str_list.append('adult')
        age_str_list.append('pediatric')
        
        sex_str_list.append('male')
        sex_str_list.append('female')
        
        loc_str_list.append('trachea')
        loc_str_list.append('left anterior chest')
        loc_str_list.append('right anterior chest')
        loc_str_list.append('left posterior chest')
        loc_str_list.append('right posterior chest')
        loc_str_list.append('left lateral chest')
        loc_str_list.append('right lateral chest')
        
        dev_str_list.append('Meditron')
        dev_str_list.append('LittC2SE')
        dev_str_list.append('Litt3200')
        dev_str_list.append('AKGC417L')
        
        ##
        age_str_list.pop(age_tmp)
        sex_str_list.pop(sex)
        loc_str_list.pop(loc)
        dev_str_list.pop(dev)
        
        age_str = random.choice(age_str_list)
        sex_str = random.choice(sex_str_list)
        loc_str = random.choice(loc_str_list)
        dev_str = random.choice(dev_str_list)

    
    if training:
        if args.meta_mode == 'age':
            meta_dict = {"age": age_str, "sex": None, "loc": None, "dev": None}
        elif args.meta_mode == 'sex':
            meta_dict = {"age": None, "sex": sex_str, "loc": None, "dev": None}
        elif args.meta_mode == 'loc':
            meta_dict = {"age": None, "sex": None, "loc": loc_str, "dev": None}
        elif args.meta_mode == 'dev':
            meta_dict = {"age": None, "sex": None, "loc": None, "dev": dev_str}
        elif args.meta_mode == 'age_sex':
            meta_dict = {"age": age_str, "sex": sex_str, "loc": None, "dev": None}
        elif args.meta_mode == 'age_loc':
            meta_dict = {"age": age_str, "sex": None, "loc": loc_str, "dev": None}
        elif args.meta_mode == 'age_dev':
            meta_dict = {"age": age_str, "sex": None, "loc": None, "dev": dev_str}
        elif args.meta_mode == 'sex_loc':
            meta_dict = {"age": None, "sex": sex_str, "loc": loc_str, "dev": None}
        elif args.meta_mode == 'sex_dev':
            meta_dict = {"age": None, "sex": sex_str, "loc": None, "dev": dev_str}
        elif args.meta_mode == 'loc_dev':
            meta_dict = {"age": None, "sex": None, "loc": loc_str, "dev": dev_str}
        elif args.meta_mode == 'age_sex_loc':
            meta_dict = {"age": age_str, "sex": sex_str, "loc": loc_str, "dev": None}
        elif args.meta_mode == 'age_sex_dev':
            meta_dict = {"age": age_str, "sex": sex_str, "loc": None, "dev": dev_str}
        elif args.meta_mode == 'age_loc_dev':
            meta_dict = {"age": age_str, "sex": None, "loc": loc_str, "dev": dev_str}
        elif args.meta_mode == 'sex_loc_dev':
            meta_dict = {"age": None, "sex": sex_str, "loc": loc_str, "dev": dev_str}
        elif args.meta_mode == 'all':
            meta_dict = {"age": age_str, "sex": sex_str, "loc": loc_str, "dev": dev_str}
        
        output_str = generate_meta_description(**meta_dict)        
        return output_str
    
    else:
        if args.test_drop_key:
            drop_key = False
            if random.random() < args.test_drop_key_prob:
                drop_key = True
            
            if args.meta_mode == 'age':
                meta_dict = {"age": 'unknown' if drop_key else age_str, "sex": None, "loc": None, "dev": None}            
            elif args.meta_mode == 'sex':
                meta_dict = {"age": None, "sex": 'unknown' if drop_key else sex_str, "loc": None, "dev": None}            
            elif args.meta_mode == 'loc':
                meta_dict = {"age": None, "sex": None, "loc": 'unknown' if drop_key else loc_str, "dev": None}            
            elif args.meta_mode == 'dev':
                meta_dict = {"age": None, "sex": None, "loc": None, "dev": 'unknown' if drop_key else dev_str}
            
            elif args.meta_mode == 'age_sex':
                if drop_key:
                    dice = random.randint(0, 1)
                    if dice == 0:
                        age_str = 'unknown'
                    else:
                        sex_str = 'unknown'
                meta_dict = {"age": age_str, "sex": sex_str, "loc": None, "dev": None}
            
            elif args.meta_mode == 'age_loc':
                if drop_key:
                    dice = random.randint(0, 1)
                    if dice == 0:
                        age_str = 'unknown'
                    else:
                        loc_str = 'unknown'
                meta_dict = {"age": age_str, "sex": None, "loc": loc_str, "dev": None}
            
            elif args.meta_mode == 'age_dev':
                if drop_key:
                    dice = random.randint(0, 1)
                    if dice == 0:
                        age_str = 'unknown'
                    else:
                        dev_str = 'unknown'
                meta_dict = {"age": age_str, "sex": None, "loc": None, "dev": dev_str}
            
            elif args.meta_mode == 'sex_loc':
                if drop_key:
                    dice = random.randint(0, 1)
                    if dice == 0:
                        sex_str = 'unknown'
                    else:
                        loc_str = 'unknown'
                meta_dict = {"age": None, "sex": sex_str, "loc": loc_str, "dev": None}
            
            elif args.meta_mode == 'sex_dev':
                if drop_key:
                    dice = random.randint(0, 1)
                    if dice == 0:
                        sex_str = 'unknown'
                    else:
                        dev_str = 'unknown'
                meta_dict = {"age": None, "sex": sex_str, "loc": None, "dev": dev_str}
            
            elif args.meta_mode == 'loc_dev':
                if drop_key:
                    dice = random.randint(0, 1)
                    if dice == 0:
                        loc_str = 'unknown'
                    else:
                        dev_str = 'unknown'
                meta_dict = {"age": None, "sex": None, "loc": loc_str, "dev": dev_str}
            
            elif args.meta_mode == 'age_sex_loc':
                if drop_key:
                    dice = random.randint(0, 2)
                    if dice == 0:
                        age_str = 'unknown'
                    elif dice == 1:
                        sex_str = 'unknown'
                    else:
                        loc_str = 'unknown'
                meta_dict = {"age": age_str, "sex": sex_str, "loc": loc_str, "dev": None}
            
            elif args.meta_mode == 'age_sex_dev':
                if drop_key:
                    dice = random.randint(0, 2)
                    if dice == 0:
                        age_str = 'unknown'
                    elif dice == 1:
                        sex_str = 'unknown'
                    else:
                        dev_str = 'unknown'
                meta_dict = {"age": age_str, "sex": sex_str, "loc": None, "dev": dev_str}
            
            elif args.meta_mode == 'age_loc_dev':
                if drop_key:
                    dice = random.randint(0, 2)
                    if dice == 0:
                        age_str = 'unknown'
                    elif dice == 1:
                        loc_str = 'unknown'
                    else:
                        dev_str = 'unknown'
                meta_dict = {"age": age_str, "sex": None, "loc": loc_str, "dev": dev_str}
            
            elif args.meta_mode == 'sex_loc_dev':
                if drop_key:
                    dice = random.randint(0, 2)
                    if dice == 0:
                        sex_str = 'unknown'
                    elif dice == 1:
                        loc_str = 'unknown'
                    else:
                        dev_str = 'unknown'
                meta_dict = {"age": None, "sex": sex_str, "loc": loc_str, "dev": dev_str}
            
            elif args.meta_mode == 'all':
                if drop_key:
                    dice = random.randint(0, 3)
                    if dice == 0:
                        age_str = 'unknown'
                    elif dice == 1:
                        sex_str = 'unknown'
                    elif dice == 2:
                        loc_str = 'unknown'
                    else:
                        dev_str = 'unknown'
                meta_dict = {"age": age_str, "sex": sex_str, "loc": loc_str, "dev": dev_str}
        
        elif args.test_unknown_all:
            return 'No description.'
        
        else:
            if args.meta_mode == 'age':
                meta_dict = {"age": age_str, "sex": None, "loc": None, "dev": None}
            elif args.meta_mode == 'sex':
                meta_dict = {"age": None, "sex": sex_str, "loc": None, "dev": None}
            elif args.meta_mode == 'loc':
                meta_dict = {"age": None, "sex": None, "loc": loc_str, "dev": None}
            elif args.meta_mode == 'dev':
                meta_dict = {"age": None, "sex": None, "loc": None, "dev": dev_str}
            elif args.meta_mode == 'age_sex':
                meta_dict = {"age": age_str, "sex": sex_str, "loc": None, "dev": None}
            elif args.meta_mode == 'age_loc':
                meta_dict = {"age": age_str, "sex": None, "loc": loc_str, "dev": None}
            elif args.meta_mode == 'age_dev':
                meta_dict = {"age": age_str, "sex": None, "loc": None, "dev": dev_str}
            elif args.meta_mode == 'sex_loc':
                meta_dict = {"age": None, "sex": sex_str, "loc": loc_str, "dev": None}
            elif args.meta_mode == 'sex_dev':
                meta_dict = {"age": None, "sex": sex_str, "loc": None, "dev": dev_str}
            elif args.meta_mode == 'loc_dev':
                meta_dict = {"age": None, "sex": None, "loc": loc_str, "dev": dev_str}
            elif args.meta_mode == 'age_sex_loc':
                meta_dict = {"age": age_str, "sex": sex_str, "loc": loc_str, "dev": None}
            elif args.meta_mode == 'age_sex_dev':
                meta_dict = {"age": age_str, "sex": sex_str, "loc": None, "dev": dev_str}
            elif args.meta_mode == 'age_loc_dev':
                meta_dict = {"age": age_str, "sex": None, "loc": loc_str, "dev": dev_str}
            elif args.meta_mode == 'sex_loc_dev':
                meta_dict = {"age": None, "sex": sex_str, "loc": loc_str, "dev": dev_str}
            elif args.meta_mode == 'all':
                meta_dict = {"age": age_str, "sex": sex_str, "loc": loc_str, "dev": dev_str}
        
        output_str = generate_meta_description(**meta_dict)
        
        if args.test_bmi:
            adult_bmi = float(metadata[2])
            child_weight = float(metadata[3])
            child_height = float(metadata[4])
            
            if adult_bmi == -1.0:
                bmi = round(child_weight / ((child_height * 0.01) ** 2), 1)
            else:
                bmi = adult_bmi
            bmi_sentence = ' The BMI of the patient was {}.'.format(bmi)
            output_str += bmi_sentence
        
        return output_str


def get_individual_cycles_torchaudio(args, recording_annotations, data_folder, filename, sample_rate, n_cls):
    sample_data = []
    fpath = os.path.join(data_folder, filename+'.wav')
    data, sr = torchaudio.load(fpath)
    
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        data = resample(data)
    
    fade_samples_ratio = 16
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
    
    for idx in recording_annotations.index:
        row = recording_annotations.loc[idx]

        start = row['Start'] # start time (second)
        end = row['End'] # end time (second)
        audio_chunk = _slice_data_torchaudio(start, end, data, sample_rate)

        if args.class_split == 'lungsound':
            crackles = row['Crackles']
            wheezes = row['Wheezes']
            label = _get_lungsound_label(crackles, wheezes, n_cls, args)
            #meta_str = get_meta_infor(metadata, args)
            
            #sample_data.append((audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls, args), meta_str))
            sample_data.append((audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls, args)))
            
        elif args.class_split == 'diagnosis':
            disease = row['Disease']            
            sample_data.append((audio_chunk, _get_diagnosis_label(disease, n_cls)))
        
    padded_sample_data = []
    for data, label in sample_data:
        data = cut_pad_sample_torchaudio(data, args) # --> resample to [1, 128000] --> 8 seconds
        padded_sample_data.append((data, label))

    return padded_sample_data


def generate_fbank(args, audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    if args.model in ['ast']:
        mean, std =  -4.2677393, 4.5689974
    else:
        mean, std = fbank.mean(), fbank.std()
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank 


# ==========================================================================


# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================
