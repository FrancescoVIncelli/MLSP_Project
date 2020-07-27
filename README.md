# Deep Learning for Speech Recognition in Humanoid Robots
Repository contenente i files del progetto dell'esame di MLSP

## 1. Dataset
Per gli esperimenti eseguiti è stato utilizzato il dataset 'Speech Commands Dataset' fornito da Tensorflow (description: https://www.tensorflow.org/datasets/catalog/speech_commands | download: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz.)
Comprendente migliaia di file audio contenneti l'espressione di brevi parole pronunciate da migliaia di persone.

## 2. Preprocessing dei file audio
Nel dataset utilizzato, i files audio hanno la durata di 1 secondo e un sampling rate molto elevato.
Il preprocessing è eseguito mediante la funzione `wav2mfcc` che produce:
- rimozione dei comandi contenuti in file audio più brevi di 1 secondo
- estrazione delle features mediante il calcolo di *Mel-frequency Cepstral coefficient* e *40-dimensional log Mel-filter-bank coefficients*

## 3. Architetture utilizzatel
Il file `models` contiene due metodi per la creazione di due modelli neurali
> `DNN_model` definisce una Deep Neural Network sulla base dell'architettura descritta nel paper 'Deep Speech 2: End-to-End Speech Recognition in
English and Mandarin' ( https://arxiv.org/pdf/1512.02595.pdf ). Il modello presenta una architettura 'mista':
- 3 Convolutional-1D layers (con Max Pooling e Batch Normalization layers e ReLU activation function)
- 3 GRU layers da 128 unità, preceduta e seguita da BAtch Normalization layers
- 1 Fully Connected layer d 256 unità

> `QNN_model` definisce una Quaternion Neural Network composta da:
- 10 1-D Quaternion Convolutional layers (con Max Pooling e PReLU activation function)
- 3 Fully Connected layer da 256 unità
( La rete è ispirata all'architettura convoluzionale descritta nel paper: https://arxiv.org/pdf/1811.09678.pdf )

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 98, 40)]          0         
_________________________________________________________________
quaternion_conv1d (Quaternio (None, 97, 32)            672       
_________________________________________________________________
p_re_lu (PReLU)              (None, 97, 32)            3104      
_________________________________________________________________
quaternion_conv1d_1 (Quatern (None, 96, 64)            1088      
_________________________________________________________________
p_re_lu_1 (PReLU)            (None, 96, 64)            6144      
_________________________________________________________________
quaternion_conv1d_2 (Quatern (None, 95, 64)            2112      
_________________________________________________________________
p_re_lu_2 (PReLU)            (None, 95, 64)            6080      
_________________________________________________________________
quaternion_conv1d_3 (Quatern (None, 94, 64)            2112      
_________________________________________________________________
p_re_lu_3 (PReLU)            (None, 94, 64)            6016      
_________________________________________________________________
quaternion_conv1d_4 (Quatern (None, 93, 128)           4224      
_________________________________________________________________
p_re_lu_4 (PReLU)            (None, 93, 128)           11904     
_________________________________________________________________
quaternion_conv1d_5 (Quatern (None, 92, 128)           8320      
_________________________________________________________________
p_re_lu_5 (PReLU)            (None, 92, 128)           11776     
_________________________________________________________________
quaternion_conv1d_6 (Quatern (None, 91, 128)           8320      
_________________________________________________________________
p_re_lu_6 (PReLU)            (None, 91, 128)           11648     
_________________________________________________________________
quaternion_conv1d_7 (Quatern (None, 90, 256)           16640     
_________________________________________________________________
p_re_lu_7 (PReLU)            (None, 90, 256)           23040     
_________________________________________________________________
quaternion_conv1d_8 (Quatern (None, 89, 256)           33024     
_________________________________________________________________
p_re_lu_8 (PReLU)            (None, 89, 256)           22784     
_________________________________________________________________
quaternion_conv1d_9 (Quatern (None, 88, 256)           33024     
_________________________________________________________________
p_re_lu_9 (PReLU)            (None, 88, 256)           22528     
_________________________________________________________________
flatten (Flatten)            (None, 22528)             0         
_________________________________________________________________
quaternion_dense (Quaternion (None, 256)               1442048   
_________________________________________________________________
quaternion_dense_1 (Quaterni (None, 256)               16640     
_________________________________________________________________
quaternion_dense_2 (Quaterni (None, 256)               16640     
_________________________________________________________________
dense (Dense)                (None, 7)                 1799      
=================================================================
Total params: 1,711,687
Trainable params: 1,711,687
Non-trainable params: 0
_________________________________________________________________
```

## Da completare
- Connectionist Temporal Classification (CTC) per migliorare il sequence-to-sequence mapping task da un segnale acustico $X = [x_1, ... , x_n]$ ad una sequenza di simboli $T = [t_1, ...,  t_m]$ (Definire CTC loss function come descritto in https://arxiv.org/pdf/1811.09678.pdf )

- Addestrare i modelli su un dataset differente (ad esempio TIMIT)
