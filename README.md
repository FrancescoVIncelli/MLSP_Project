# Deep Learning for Speech Recognition in Humanoid Robots
Repository contenente i files del progetto dell'esame di MLSP

## 1. Dataset
Per gli esperimenti eseguiti è stato utilizzato il dataset 'Speech Commands Dataset' fornito da Tensorflow (description: https://www.tensorflow.org/datasets/catalog/speech_commands | download: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz.)
Comprendente migliaia di file audio contenneti l'espressione di brevi parole pronunciate da migliaia di persone.

## 2. Preprocessing dei file audio
Nel dataset utilizzato, i files audio hanno la durata di 1 secondo e un sampling rate molto elevato.
Il preprocessing è eseguito mediante la funzione `wav2mfcc` che produce:
- rimozione dei comandi contenuti in file audio più brevi di 1 secondo
- estrazione delle features mediante il calcolo di *Mel-frequency Cepstral coefficient* e *40-dimensional log Mel-filter-bank coefficients* (seguendo il seguente blog: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html )
- Creazione della matrice di input quaternionica (per il training della rete QNN ) 

## 3. Architetture utilizzatel
Il file `models` contiene due metodi per la creazione di due modelli neurali
> `DNN_model` definisce una Deep Neural Network sulla base dell'architettura descritta nel paper 'Deep Speech 2: End-to-End Speech Recognition in
English and Mandarin' ( https://arxiv.org/pdf/1512.02595.pdf ). Il modello presenta una architettura 'mista':
- 3 Convolutional-1D layers (con Max Pooling e Batch Normalization layers e ReLU activation function)
- 3 GRU layers da 128 unità, preceduta e seguita da BAtch Normalization layers
- 1 Fully Connected layer d 256 unità

![alt text](https://github.com/FrancescoVIncelli/MLSP_Project/blob/master/images/dnn_model_architecture.png)

> `QNN_model` definisce una Quaternion Neural Network composta da:
- 4 Quaternion Convolutional 1D layers (con Max Pooling e PReLU activation function)
- 3 Fully Connected layer da 256 unità
( La rete è ispirata all'architettura convoluzionale descritta nel paper: https://arxiv.org/pdf/1811.09678.pdf )

![alt text](https://github.com/FrancescoVIncelli/MLSP_Project/blob/master/images/qnn_model_architecture.png)

## Risultati
Entrambi i modelli sono stati trainati su 50 epoche, con un sottoinsieme di sette *command words* del dataset utilizzato
> DNN (Non-Quaternion) model
Epoch 50/50 | loss: 0.1522 - accuracy: 0.9481 - val_loss: 0.1576 - val_accuracy: 0.9500

> Quaternion-NN model
Epoch 50/50 | loss: 0.0552 - accuracy: 0.9900 - val_loss: 0.8690 - val_accuracy: 0.8567

![alt text](https://github.com/FrancescoVIncelli/MLSP_Project/blob/master/images/qnn_train_val_acc.png)

![alt text](https://github.com/FrancescoVIncelli/MLSP_Project/blob/master/images/qnn_train_val_loss.png)
## Da completare
- Connectionist Temporal Classification (CTC) per migliorare il sequence-to-sequence mapping task da un segnale acustico X = [x_1, ... , x_n] ad una sequenza di simboli T = [t_1, ...,  t_m] (Definire CTC loss function come descritto in https://arxiv.org/pdf/1811.09678.pdf )

- Addestrare i modelli su un dataset differente (ad esempio TIMIT)
