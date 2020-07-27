# Deep Learning for Speech Recognition in Humanoid Robots
Repository contenente i files del progetto dell'esame di MLSP

## 1. Dataset
Per gli esperimenti eseguiti è stato utilizzato il dataset 'Speech Commands Dataset' fornito da Tensorflow (description: https://www.tensorflow.org/datasets/catalog/speech_commands | download: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz.)
Comprendente migliaia di file audio contenneti l'espressione di brevi parole pronunciate da migliaia di persone.

## 2. Preprocessing dei file audio
Nel dataset utilizzato, i files audio hanno la durata di 1 secondo e un sampling rate molto elevato.
Il preprocessing è eseguito mediante la funzione `wav2mfcc` che produce:
> rimozione dei comandi contenuti in file audio più brevi di 1 secondo
> estrazione delle features mediante il calcolo di *Mel-frequency Cepstral coefficient* e *40-dimensional log Mel-filter-bank coefficients*

## 3. Architetture utilizzatel
Il file `models` contiene due metodi per la creazione di due modelli neurali
> `DNN_model` definisce una Deep Neural Network sulla base dell'architettura descritta nel paper 'Deep Speech 2: End-to-End Speech Recognition in
English and Mandarin' ( https://arxiv.org/pdf/1512.02595.pdf ). L'architettura è composta da due sub-networks:
>> 1) una CNN composta da tre 1-D Convolutional layers (con Max Pooling e Batch Normalization layers)
>> 2) una RNN composta da 3 GRU layers da 128 unità, preceduta e seguita da BAtch Normalization layers
>> 3) un Fully Connected layer d 256 unità

> `QNN_model` definisce una Quaternion Neural Network composta da 10 1-D Quaternion Convolutional layers (con Max Pooling e PReLU activation function) e 3 Fully Connected layer da 256 unità
( La rete è ispirata all'architettura convoluzionale descritta nel paper: https://arxiv.org/pdf/1811.09678.pdf )

## Da completare
- Connectionist Temporal Classification (CTC) per migliorare il sequence-to-sequence mapping task da un segnale acustico $X = [x_1, ... , x_n]$ ad una sequenza di simboli $T = [t_1, ...,  t_m]$ (Definire CTC loss function come descritto in https://arxiv.org/pdf/1811.09678.pdf )

- Addestrare i modelli su un dataset differente (ad esempio TIMIT)
