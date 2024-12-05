# BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound Classification (INTERSPEECH 2024)
[arXiv](https://arxiv.org/abs/2406.06786) | [Conference](https://www.isca-archive.org/interspeech_2024/kim24f_interspeech.html) | [BibTeX](#bibtex)

<p align="center">
<img width="1057" alt="image" src="https://github.com/user-attachments/assets/afb5d3cb-2ce6-4e31-9fcf-ab44c37da22d">
</p>


Official Implementation of **BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound Classification.**<br/>

**See you in INTERSPEECH 2024!**

## Prerequisites
Please check environments and requirements before you start. If required, we recommend you to either upgrade versions or install them for smooth running.

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

### Environments
`Ubuntu xx.xx`  
`Python 3.8.xx`

## Environmental set-up

Install the necessary packages with:

run `requirements.txt`

```
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

For the reproducibility, we used torch=2.0.1+cu117 and torchaudio=2.0.1+cu117, so we highly recommend install as follow:

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

## Datasets
Download the ICBHI files and unzip it.
All details is described in the [paper w/ code](https://paperswithcode.com/dataset/icbhi-respiratory-sound-database)

```
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
or 
wget --no-check-certificate https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
```

All `*.wav` and `*.txt` should be saved in data/icbhi_dataset/audio_test_data. (i.e., mkdir `audio_test_data` into `data/icbhi_dataset/` and move `*.wav` and `*.txt` into `data/icbhi_dataset/audio_test_data/`)

Note that ICBHI dataset consists of a total of 6,898 respiratory cycles, of which 1,864 contain crackles, 886 contain wheezes, and 506 contain both crackles and wheezes, in 920 annotated audio samples from 126 subjects.

## Run

### Audio-CLAP for Respiratory Sound Classification
```
$ ./scripts/icbhi_audio-clap_ce.sh
```

### BTS for Respiratory Sound Classification
```
$ ./scripts/icbhi_bts_meta_all.sh
```

### Evaluation with BTS for Respiratory Sound Classification
```
$ ./scripts/eval_bts.sh
```
Note that change `--pretrained_ckpt` with your directory. (e.g. `--pretrained_ckpt /home2/jw/workspace/crisp/save/icbhi_laion/clap-htsat-unfused_ce_bs8_lr5e-5_ep50_seed1_check2/best.pth`)


We will provide pretrained checkpoint into the camera-ready version

## ICBHI Data

The database consists of a total of 5.5 hours of recordings containing 6898 respiratory cycles, of which 1864 contain crackles, 886 contain wheezes, and 506 contain both crackles and wheezes, in 920 annotated audio samples from 126 subjects.

The downloaded data looks like [[kaggle](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database), [paper w/ code](https://paperswithcode.com/dataset/icbhi-respiratory-sound-database)]:

<pre>
data/icbhi_dataset
├── metadata.txt
│    ├── Patient number
│    ├── Age
│    ├── Sex
│    ├── Adult BMI (kg/m2)
│    ├── Adult Weight (kg)
│    └── Child Height (cm)
│
├── official_split.txt
│    ├── Patient number_Recording index_Chest location_Acqiosotopm mode_Recording equipment
│    |    ├── Chest location
│    |    |    ├── Trachea (Tc),Anterior left (Al),Anterior right (Ar),Posterior left (Pl)
│    |    |    └── Posterior right (Pr),Lateral left (Ll),Lateral right (Lr)
│    |    |
│    |    ├── Acquisition mode
│    |    |    └── sequential/single channel (sc), simultaneous/multichannel (mc)
│    |    |
│    |    └── Recording equipment 
│    |         ├── AKG C417L Microphone (AKGC417L), 
│    |         ├── 3M Littmann Classic II SE Stethoscope (LittC2SE), 
│    |         ├── 3M Litmmann 3200 Electronic Stethoscope (Litt3200), 
│    |         └── WelchAllyn Meditron Master Elite Electronic Stethoscope (Meditron)
│    |    
│    └── Train/Test   
│
├── patient_diagnosis.txt
│    ├── Patient number
│    └── Diagnosis
│         ├── COPD: Chronic Obstructive Pulmonary Disease
│         ├── LRTI: Lower Respiratory Tract Infection
│         └── URTI: Upper Respiratory Tract Infection
│
└── patient_list_foldwise.txt
</pre>

## Result
The proposed BTS achieves a 63.54% Score, which is the new state-of-the-art performance in ICBHI score.
<p align="center">
<img width="1072" alt="image" src="https://github.com/user-attachments/assets/dd415f4c-6f04-4713-9d27-e6cbee503f04">
</p>



## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{kim24f_interspeech,
  title     = {BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound Classification},
  author    = {June-Woo Kim and Miika Toikkanen and Yera Choi and Seoung-Eun Moon and Ho-Young Jung},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {1690--1694},
  doi       = {10.21437/Interspeech.2024-492},
  issn      = {2958-1796},
}
```
