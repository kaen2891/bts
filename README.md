# BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound Classification (INTERSPEECH 2024)
[arXiv](https://arxiv.org/abs/2406.06786) | [BibTeX](#bibtex)

<p align="center">
<img width="704" alt="image" src="https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/assets/46586785/d6f51658-df38-4a8b-8174-0cca9c8127fa">
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
<img width="696" alt="image" src="https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/assets/46586785/0cdeca6c-89b0-4cba-ab9f-4307f54c2c8d">
</p>



## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@article{kim2024bts,
  title={BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound Classification},
  author={Kim, June-Woo and Toikkanen, Miika and Choi, Yera and Moon, Seoung-Eun and Jung, Ho-Young},
  journal={arXiv preprint arXiv:2406.06786},
  year={2024}
}
```