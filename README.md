# Deep Learning for Speech Recognition in Humanoid Robots
Repository contenente i files del progetto dell'esame di MLSP

# Aggiornamenti
- Eseguita la correzione della creazione della *input features matrix* e delle *target variables* dai file audio del dataset TIMIT
- Definita una classe per la creazione del modello neurale convoluzionale quaternionico e non quaternionico, con specifica degli iperparametri dei layers
- Definita una funzione per la coversione del tipo dei file audio in TIMIT dataset da NIST SPHERE a .wav

> Aggiornato il preprocessamento del dataset TIMTI per phoneme recognition mediante Quaternion Neural Networks
> Aggiunti modelli quaternionici per *phoneme recognition* task: interspeech_model e QCNN_model (ottenuto modificando il modello precedente)
> Aggiunta la Connectionist Temporal Classification (CTC) per migliorare il sequence-to-sequence mapping task da un segnale acustico X = [x_1, ... , x_n] ad una sequenza di simboli T = [t_1, ...,  t_m] (CTC implementata come descritto in https://arxiv.org/pdf/1811.09678.pdf )

# Esperimenti su TIMIT Dataset
## 1. Dataset
[TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/LDC93S1) , contenente registrazioni di 630 speakers ricondicibili agli otto principali dialetti dell'inglese americano, comprendenti frasi foneticamente ricche.

## 2. Preprocessing dei file audio
I file audio presenti nel dataset hanno durante diverse in un range che viara da 0.Xs a 5.Xs . Sui file audio più brevi è stata quindi effettuata una procedura di padding, aggiungendo alla fine della sequenza campionata un segnale nullo. Quindi, è stato eseguita l'estrazione delle features, calcolando i *40-dimensional log Mel-filter-bank coefficients*, madiante le funzioni della libreria `python_speech_features`. La matrice di input quaternionica è stata quindi ottenuta seguendo la procedura descritta in [Speech Recognition with Quaternion Neural Networks](https://arxiv.org/abs/1811.09678) componendo un vettore quaternionico cone le derivate prima, seconda e terza del filter-bank coefficients vector.

## 3. Architetture utilizzate
Nel file `asr_models` sono definite più classi di modelli neurali
> `simple_cnn_1D` e `simple_cnn_2D` definiscono una rispettivamente una architettura 1D e 2D convoluzionale che riprende l'architettura descritta nel paper [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf). Il modello presenta una architettura 'mista':
- *N_conv* Conv1D o Conv-2D layers (con *N_conv* in {3,4,5,6}). Ciascun layer convoluzionale è seguito da un activation function layer *PReLU* , quindi l'output dell'intero blocco di *N_conv* convolutional layers è passato in input ad un layer per la Batch-Normalization.
- *N_dense* Fully-Connected layers seguono il blocco convoluzionale (con *N_dense* in {1,2,3}), ciascuno avente 1024 unità. Ciascun dense layer è inglobato in un *TimeDistributed* layer per applicare l'output del layer a ciascun intervallo temporale dell'input. Eventualmente può essere applicato dropout con probabilità 0.3 per ridurre l'overfitting. Anche il *Blocco Fully-Connected* è seguito da un layer per la Batch-Normalization.
- Un dense layer finale, con numero di unità pari al numero di classi delle dell'insieme di fonemi, genera l'output del modello

> `sequrntial_cnn_1D` e `sequrntial_cnn_1D` sono modelli sequenziali che presentano un'architettura analoga ai modelli `simple_cnn_1D` e `simple_cnn_2D` , ma a differenza di questi ultimi, essi sono definiti per essere addestrati sulla base della categoical cross-entropy loss (e non la CTC loss).
In questo caso, il valore iniziale della loss è molto più basso rispetto ai rispettivi modelli che adottano la CTC loss (~140 vs. ~0.72) , e anche l'overfitting è minore.

## Risultati
### `simple_cnn_1D` Model
* Model architecture summary
```
Model: "simple_cnn_1D"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       [(None, 778, 160)]        0         
_________________________________________________________________
batch_normalization_10 (Batc (None, 778, 160)          640       
_________________________________________________________________
conv_0 (Conv1D)              (None, 778, 8)            3848      
_________________________________________________________________
p_re_lu_43 (PReLU)           (None, 778, 8)            6224      
_________________________________________________________________
conv_1 (Conv1D)              (None, 778, 16)           400       
_________________________________________________________________
p_re_lu_44 (PReLU)           (None, 778, 16)           12448     
_________________________________________________________________
conv_2 (Conv1D)              (None, 778, 32)           1568      
_________________________________________________________________
p_re_lu_45 (PReLU)           (None, 778, 32)           24896     
_________________________________________________________________
conv_3 (Conv1D)              (None, 778, 64)           6208      
_________________________________________________________________
p_re_lu_46 (PReLU)           (None, 778, 64)           49792     
_________________________________________________________________
conv_4 (Conv1D)              (None, 778, 128)          24704     
_________________________________________________________________
p_re_lu_47 (PReLU)           (None, 778, 128)          99584     
_________________________________________________________________
conv_5 (Conv1D)              (None, 778, 256)          98560     
_________________________________________________________________
p_re_lu_48 (PReLU)           (None, 778, 256)          199168    
_________________________________________________________________
batch_normalization_11 (Batc (None, 778, 256)          1024      
_________________________________________________________________
time_distributed_15 (TimeDis (None, 778, 1024)         263168    
_________________________________________________________________
p_re_lu_49 (PReLU)           (None, 778, 1024)         796672    
_________________________________________________________________
time_distributed_16 (TimeDis (None, 778, 1024)         1049600   
_________________________________________________________________
p_re_lu_50 (PReLU)           (None, 778, 1024)         796672    
_________________________________________________________________
time_distributed_17 (TimeDis (None, 778, 1024)         1049600   
_________________________________________________________________
p_re_lu_51 (PReLU)           (None, 778, 1024)         796672    
_________________________________________________________________
batch_normalization_12 (Batc (None, 778, 1024)         4096      
_________________________________________________________________
dense_20 (Dense)             (None, 778, 61)           62525     
_________________________________________________________________
softmax (Activation)         (None, 778, 61)           0         
=================================================================
Total params: 5,348,069
Trainable params: 5,345,189
Non-trainable params: 2,880
_________________________________________________________________
```

* Train session(s)
Il modello è stato trainato sun un numero totale di epoche di 50, con i seguenti valori finali di CTC loss sul training e sul validation set:
loss | val_loss
------------ | -------------
110.2284 | 133.7241

### `simple_cnn_2D` Model

### `sequential_cnn_1D` Model

<!--
# Esperimenti su Tensorflow Speech Commands Dataset
## 1. Dataset
Per gli esperimenti eseguiti è stato utilizzato il dataset 'Speech Commands Dataset' fornito da Tensorflow (description: https://www.tensorflow.org/datasets/catalog/speech_commands | download: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz.)
Comprendente migliaia di file audio contenneti l'espressione di brevi parole pronunciate da migliaia di persone.

<!--
## 2. Preprocessing dei file audio
Nel dataset utilizzato, i files audio hanno la durata di 1 secondo e un sampling rate molto elevato.
Il preprocessing è eseguito mediante la funzione `wav2mfcc` che produce:
- rimozione dei comandi contenuti in file audio più brevi di 1 secondo
- estrazione delle features mediante il calcolo di *Mel-frequency Cepstral coefficient* e *40-dimensional log Mel-filter-bank coefficients* (seguendo il seguente blog: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html )
- Creazione della matrice di input quaternionica (per il training della rete QNN ) 

<!--
## 3. Architetture utilizzate
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

<!--
## Risultati
Entrambi i modelli sono stati trainati su 50 epoche, con un sottoinsieme di sette *command words* del dataset utilizzato
> DNN (Non-Quaternion) model | oss: 0.1522 - accuracy: 0.9481 - val_loss: 0.1576 - val_accuracy: 0.9500

![alt text](https://github.com/FrancescoVIncelli/MLSP_Project/blob/master/images/dnn_train_val_acc.png)

![alt text](https://github.com/FrancescoVIncelli/MLSP_Project/blob/master/images/dnn_train_val_loss.png)

> Quaternion-NN model | loss: 0.0552 - accuracy: 0.9900 - val_loss: 0.8690 - val_accuracy: 0.8567

![alt text](https://github.com/FrancescoVIncelli/MLSP_Project/blob/master/images/qnn_train_val_acc.png)

![alt text](https://github.com/FrancescoVIncelli/MLSP_Project/blob/master/images/qnn_train_val_loss.png)

Il modello quaternionico ottiene risultati leggermente inferiori al modello non quaternionico. Per migliorare le prestazioni, vorrei provare ad aggiungere dei *recurrent layers* anche al modello quaternionico, anche se la libreria utilizzata (implementata in Tensorflow) non dispone di tali layers, disponibili invece nella seconda libreria implementata in PyTorch. Inoltre altre combinazioni di layers convoluzionali e alte tecniche di pre-processamento della matrice in input alla rete quaternionica sono in fase di sperimentazione.
