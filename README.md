
# Generating Random Tweets Using LSTM Neural Network

## Introduction: A Bird's Eye View

In this project, we set out to generate random tweets that will be similar to the Twitter account of Dr. Meni Adler, which can be found [here](https://twitter.com/meniadler).

To achieve this, we used a specific flavor of a Recurrent Neural Network called LSTM (Long Short Term Memory). The reason we chose this type of network is due to a problem in vanilla RNN called "vanishing gradient". What this means, in simple terms, is that vanilla RNNs can't "remember" too far back in time. LSTM solves this problem by using a different nonlinearity. A visualization of the vanishing gradient problem can be found [here](imgur.com/gallery/vaNahKE) (source: Andrej Karpathy).

As you will no doubt notice, most of the tweets are in Hebrew. This fact might've hindered our progress, since there are barely (if any) good NLP libraries for Hebrew. Luckily, our network works on characters, not on words, which means that language has no effect on the output.

As a sidenote, we added a verbosity flag to make things seem to work. In this notebook, we'll send the `verbose` variable, defined below, to each function. You, the reader, can turn it on (1) or off (0).


```python
verbose = 1
```

## Part I: Libraries

This project uses five libraries:
1. [Keras](https://github.com/fchollet/keras) - A convenient front-end for Theano. It makes building the model very simple, elegant, and coherent.
2. [Theano](http://deeplearning.net/software/theano/) - Although not made specifically for deep-learning, Theano is well-known library in the machine learning community to optimize calculations involving matrices. Part of its charm is its utilization of the GPU for its calculations.
3. [scikit-learn](http://scikit-learn.org/) - Part of the SciPy family, scikit-learn features many modules for different machine learning tasks.
4. [NumPy](http://www.numpy.org/) - Using arrays as its primary data structure, NumPy runs all its array computations in C, making it essential for any high-performance calculations in Python.
5. [Python Twitter](https://github.com/bear/python-twitter) - We chose this library for no reason other than that it was the first hit on Google for "python twitter". This library provides a very simple and Pythonic front-end for the Twitter API.

Note: Line 1 in the code below is used for compatibility with Python 3.


```python
from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances

import numpy as np
import twitter
```

    Using Theano backend.
    

## Part II: Compiling the Tweet Corpus

### Getting the Tweets

In order to obatain Dr. Adler's tweets, we use credentials from a Twitter application made solely for this purpose.
Since Twitter only allows 100 API hits per hour, we loop 100 times, each time taking the next 200 tweets. We return a list of only the text of the tweets (since the tweet object contains more data, such as screen name and id), and save it as a NumPy array on disk.

Naturally, we only had to run this code once, and from then on, only load the data we saved.


```python
def get_tweets():
    api = twitter.Api(consumer_key='qVNR9I00rQv8nLYqG51od2wRA',
                      consumer_secret='N2MYDaABeqlXMfxDMOoCbY4pGfqMUS5vWtqP4FtCPuXzo7vK6h',
                      access_token_key='3845193803-1HEnTxQLnoBqtipK1SHQ9nORmfQliQgeJzgylKx',
                      access_token_secret='EZ4NwBMy23G0d0G8Yxmmt7KfaEfZxD9vXlFziMEkPGB4U')
    tweets = []
    max_id = None
    for _ in range(100):
        tweets.extend(list(api.GetUserTimeline(screen_name='meniadler',
                                               max_id=max_id,
                                               count=200,
                                               include_rts=False,
                                               exclude_replies=True)))
        max_id = tweets[-1].id
    return [tweet.text for tweet in tweets]

np.save(file='meni_tweets.npy', arr=get_tweets())
```

### Creating the Corpus

We load the file that was created above, and filter two specific things that have shown to be troublesome along the way:
* Links - by removing any tweet containing the string `http`
* A sanitized '>' symbol - for some reason, some tweets have the sanitized '>' symbol: `&gt;`. This makes the network use this nonsensical string, which is unwanted.

The corpus is then created by joining all tweets with a space.


```python
CORPUS_LENGTH = None

def get_corpus(verbose=0):
    tweets = np.load('meni_tweets.npy')
    tweets = [t for t in tweets if 'http' not in t]
    tweets = [t for t in tweets if '&gt' not in t]
    corpus = u' '.join(tweets)
    global CORPUS_LENGTH
    CORPUS_LENGTH = len(corpus)
    if verbose:
        print('Corpus Length:', CORPUS_LENGTH)
    return corpus

corpus = get_corpus(verbose=verbose)
```

    Corpus Length: 134481
    

## Part III: Converting the Corpus

### Character $\leftrightarrow$ Index Mapping

We are going to need to One-Hot encode (explained later) our sequences. Later, when we want to generate tweets, we will need to decode our One-Hot vectors. For this purpose, we create two mappings: from a character to an index, and vice versa. We also return the set of all characters in the corpus.


```python
N_CHARS = None

def create_index_char_map(corpus, verbose=0):
    chars = sorted(list(set(corpus)))
    global N_CHARS
    N_CHARS = len(chars)
    if verbose:
        print('No. of unique characters:', N_CHARS)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

chars, char_to_idx, idx_to_char = create_index_char_map(corpus, verbose=verbose)
```

    No. of unique characters: 130
    

### Sequence Creation

Since we are working on a character-based network, we have to divide our corpus into `sequences` of equal lengths. This division decides how many training examples we will have. The example in the Keras documentation works with a sequence length of 40 and redundancy of 37 characters (3-character step). After experimenting with different values, such as a 1-character step, or longer/shorter sequences, we've concluded that the effect is either worse, or pretty much the same. For example, a 1-character step and a short sequence length results in many more training examples, making the training time much longer. Therefore, we stayed with the values from the Keras docs.

Another important thing to note is the `next_chars` variable, which is our outputs for the training examples.

The `create_sequences` function below returns our input (`sequences`) and output (`next_chars`) in raw form. We will take care of that next.


```python
MAX_SEQ_LENGTH = 40
SEQ_STEP = 3
N_SEQS = None

def create_sequences(corpus, verbose=0):
    sequences, next_chars = [], []
    for i in range(0, CORPUS_LENGTH - MAX_SEQ_LENGTH, SEQ_STEP):
        sequences.append(corpus[i:i + MAX_SEQ_LENGTH])
        next_chars.append(corpus[i + MAX_SEQ_LENGTH])
    global N_SEQS
    N_SEQS = len(sequences)
    if verbose:
        print('No. of sequences:', len(sequences))
    return np.array(sequences), np.array(next_chars)

sequences, next_chars = create_sequences(corpus, verbose=verbose)
```

    No. of sequences: 44814
    

### One-Hot Encoding

At this point, we wish to take our raw input and output, and convert them to simple binary vectors. This is done using what is called "One-Hot encoding". This means that given a corpus of length $n$, every character $c_{k}$ will have an index $i_{k}$, and will be encoded by a $n \times 1$ vector $v$ in which $v_{k} = 1$ and the rest of the entries are 0.

For the next part of the explanation, let's define some notation:
* $S = $ Number of sequences (`N_SEQS`)
* $L = $ Length of a sequence (`MAX_SEQ_LENGTH`)
* $N = $ Number of unique characters in the corpus (`N_CHARS`)

Our input, $X$, is a 3D matrix of shape $S \times L \times N$, because we $S$ training exampels, each example is $L$ One-Hot vectors of length $N$.
The output, $y$, is a 2D matrix of shape $S \times N$ which is just $S$ One-Hot encoded vectors that encode the next character after the respective sequence in $X$.

That was a lot of talk, let's see an example:
Suppose we have a corpus which has $N = 4$ unique characters - 'a', 'b', 'c', and 'd' - and we decide that $L = 3$.
The first input and output created from the string 'cabd' would look like this:
\begin{align*}
X_{0} &= \begin{bmatrix}
            0 & 1 & 0\\
            0 & 0 & 1\\
            1 & 0 & 0\\
            0 & 0 & 0
         \end{bmatrix}%
,& y_{0} &= \begin{bmatrix}
                0\\
                0\\
                0\\
                1
            \end{bmatrix}
\end{align*}


```python
def one_hot_encode(sequences, next_chars, char_to_idx):
    X = np.zeros((N_SEQS, MAX_SEQ_LENGTH, N_CHARS), dtype=np.bool)
    y = np.zeros((N_SEQS, N_CHARS), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1
    return X, y

X, y = one_hot_encode(sequences, next_chars, char_to_idx)
```

## Part IV: The Model

The model for our neural network has undergone many generations until we finally found the sweet spot. We began with a simple 3-layer network: input, LSTM, and output, with the size of the LSTM layer being 128 neurons. It was quick to train (especially on a GPU instance on AWS), and the log loss was pretty low after not too many epochs, but the results themselves were not satisfactory.

We tried increasing the number of neurons in the hidden layer from 128 to 256 and 512. With 256, there was no notable improvement, and at 512, the training time lengthened significantly, and the log loss was much higher. This is probably because the tweet corpus is not very large, and using too many neurons was simply "too much".

Following the attempts above (and trying other, larger, datasets for comparison), we decided to use a 4-layer model with a 20% dropout (which seems to be the recommended value). Two LSTM layers at 128 neurons each gave a good trade-off between the log loss, training time, and the end results.

A note about the `loss` (line 8), `activation` (line 7), and `optimizer` (line 8) parameters: we use the [softmax](https://en.wikipedia.org/wiki/Softmax_function) activation to output a probablity distribution across the characters. Later, when we sample the characters, we'll take the one with the highest probability of being the next one. `categorical_crossentropy` means that our loss function is the [log loss](https://en.wikipedia.org/wiki/Cross_entropy) function, the same one used in logistic regression. This makes sense, since we can look at sequence generation as a classification problem. Regarding the optimizer, we could just write `optimizer='rmsprop'`, but this uses the RMSProp optimizer (recommended by the Keras docs as a good choice for RNNs) with a learning rate of $0.1$. We later found that a learning rate of $0.01$ performs better.


```python
def build_model(hidden_layer_size=128, dropout=0.2, learning_rate=0.01, verbose=0):
    model = Sequential()
    model.add(LSTM(hidden_layer_size, return_sequences=True, input_shape=(MAX_SEQ_LENGTH, N_CHARS)))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_layer_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(N_CHARS, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate))
    if verbose:
        print('Model Summary:')
        model.summary()
    return model

model = build_model(verbose=verbose)
```

    Model Summary:
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    lstm_7 (LSTM)                    (None, 40, 128)       132608      lstm_input_4[0][0]               
    ____________________________________________________________________________________________________
    dropout_7 (Dropout)              (None, 40, 128)       0           lstm_7[0][0]                     
    ____________________________________________________________________________________________________
    lstm_8 (LSTM)                    (None, 128)           131584      dropout_7[0][0]                  
    ____________________________________________________________________________________________________
    dropout_8 (Dropout)              (None, 128)           0           lstm_8[0][0]                     
    ____________________________________________________________________________________________________
    dense_4 (Dense)                  (None, 130)           16770       dropout_8[0][0]                  
    ====================================================================================================
    Total params: 280962
    ____________________________________________________________________________________________________
    

## Part V: Training the Model

This is the part where there's sometimes enough time to go to sleep, have a snack, bake a cake, save the world, etc. The worst part was that the models that were the slowest to train ended up being.. hell, I'll say it: they were shit. 6 hours, 11 hours of training, and it sucked.

The model we ended up with (described above) took ~30 seconds per epoch, and we trained it for more than a 100 epochs (either 120 or 180, we don't remember, it was late).

On line 2, we see the lifesaver from Keras - the ModelCheckpoint callback. Every epoch, the `checkpointer` sees if the model's parameters improved - if they did, it saves them to a file called `weights.hdf5`.

We kept the batch size at 128 since we didn't see a reason to increase or decrease it, really.


```python
def train_model(model, X, y, batch_size=128, nb_epoch=60, verbose=0):
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='loss', verbose=verbose, save_best_only=True, mode='min')
    model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[checkpointer])

train_model(model, X, y, verbose=verbose)
```

## Part VI: Generating New Tweets

OK, here comes the fun part! Generating new tweets from our trained network! Since the following code will have a lot of random choices, we first set a seed for reproducibility.


```python
np.random.seed(1337)
```

We start by defining a function which picks the next character based on the probablity distribution. On line 3 we use a 'diversity' value of $0.2$. This value decides how "wild" the sampling would be (so $1$, for example, is completely random).
We went over the outputs for different values, and saw that at $0.2$ the output makes the most sense, to us. The tweets read better, and sounded more like a drunk Dr. Adler than a computer-generated tweet.


```python
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / 0.2
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

Let's generate some tweets. First, we load the weights that were output as part of the training process. That way, we could train the network overnight, copy the weights to our own computers, and turn off the AWS instance.
To make things easier for us to read, we decided that the seed would be a sequence beginning with a space. This is not an optimization in any way, and is only useful for the human reader.

A new tweet is generated in the following way:
1. A random sequence seed is chosen. This will be the beginning of the tweet. Since a tweet is 140 characters, and we already have 40 from the seed, we wish to generate 100 more characters.
2. Input $x$ is created to feed to the network. The input is built by One-Hot encoding the last seen sequence.
3. The model predicts a probability distribution. Only the distribution for the next character is used.
4. The `sample` function, defined above, is used to give us the vector of the next character.
5. The next character is appended to the tweet, and the sequence seed drops its first character and adds the next.


```python
def generate_tweets(model, corpus, char_to_idx, idx_to_char, n_tweets=10, verbose=0): 
    model.load_weights('weights.hdf5')
    tweets = []
    spaces_in_corpus = np.array([idx for idx in range(CORPUS_LENGTH) if corpus[idx] == ' '])
    for i in range(1, n_tweets + 1):
        begin = np.random.choice(spaces_in_corpus)
        tweet = u''
        sequence = corpus[begin:begin + MAX_SEQ_LENGTH]
        tweet += sequence
        if verbose:
            print('Tweet no. %03d' % i)
            print('=' * 13)
            print('Generating with seed:')
            print(sequence)
            print('_' * len(sequence))
        for _ in range(100):
            x = np.zeros((1, MAX_SEQ_LENGTH, N_CHARS))
            for t, char in enumerate(sequence):
                x[0, t, char_to_idx[char]] = 1.0

            preds = model.predict(x, verbose=0)[0]
            next_idx = sample(preds)
            next_char = idx_to_char[next_idx]

            tweet += next_char
            sequence = sequence[1:] + next_char
        if verbose:
            print(tweet)
            print()
        tweets.append(tweet)
    return tweets

tweets = generate_tweets(model, corpus, char_to_idx, idx_to_char, verbose=verbose)
```

    Tweet no. 001
    =============
    Generating with seed:
     נתקעים או מקרטעים, גם כן טכנולוגיה המין
    ________________________________________
     נתקעים או מקרטעים, גם כן טכנולוגיה המין המוסכת המוצא היה מיש לא ישאר אתב של היה מישראל בשם המשמה היום לקרוא המוציא המוצא שהיה לו את המועד ה
    
    Tweet no. 002
    =============
    Generating with seed:
     לג,ב). יש המציעים לגרוס: "ממריבת קדש" -
    ________________________________________
     לג,ב). יש המציעים לגרוס: "ממריבת קדש" - היה לו אותי היה לו כי מך המקור הארץ ישראל ויש מתלים המילה והשוקל של מועד היה אינו לו אותי את המופר 
    
    Tweet no. 003
    =============
    Generating with seed:
     מדווחת שהיה אתמול שמח מאוד ברחובות באר 
    ________________________________________
     מדווחת שהיה אתמול שמח מאוד ברחובות באר של משמו בארץ יש מתרגום המשמה של היהודים במספר האני השבעים מתרגם מאות באותי המוכל המוצר המופלע במקום 
    
    Tweet no. 004
    =============
    Generating with seed:
     אחד השיחים... כי אמרה אל אראה במות הילד
    ________________________________________
     אחד השיחים... כי אמרה אל אראה במות הילדון המוצא בשות של היה לו כי משנה במקום המשמש המוצר המילה בית המקרא המוצא את המפות השוק היה לא משמשון 
    
    Tweet no. 005
    =============
    Generating with seed:
     מה גזרו על סנדל מסומר? שהיו רואות את רא
    ________________________________________
     מה גזרו על סנדל מסומר? שהיו רואות את ראש בארץ ישראל של משה במילה והיה מצואת המוצא להיה משבעים שיש מאה אינם אינה לא מוצא אותו אותו מוצא אין 
    
    Tweet no. 006
    =============
    Generating with seed:
     כיצד הפך 'העם התרבותי באירופה' לכנופיית
    ________________________________________
     כיצד הפך 'העם התרבותי באירופה' לכנופיית המילה וישראל בית המועד במקרא באותי במקרא במקום המופלא לא משמש של משפט המוצר המשמה את המועד המופל של
    
    Tweet no. 007
    =============
    Generating with seed:
     "אני הייתי בליכוד כשהיית פתק בכותל" וזה
    ________________________________________
     "אני הייתי בליכוד כשהיית פתק בכותל" וזה בית המועד המוצא היה את המועד המוצא היה לי בשב על המלאך משמש להיה מוא של היהודים של היהודים והיה מתק
    
    Tweet no. 008
    =============
    Generating with seed:
     פטפטן.
    (לופז לא מפנה למקור, אך נדמה לי 
    ________________________________________
     פטפטן.
    (לופז לא מפנה למקור, אך נדמה לי משמש להם בירושלים, והיא משב להם המשפט לשונות המוצא את המשפט ולא יותר כי משנה המילה בית המילה והמופלע
    
    Tweet no. 009
    =============
    Generating with seed:
     'רשימות בנושא הבדווים', מוקלדות במכונת 
    ________________________________________
     'רשימות בנושא הבדווים', מוקלדות במכונת היה מציין של מעשה לאותו בין הישר המוציא לא בכל השמחון להיה לבית המועד היה לו כי המשפט בו של משול המו
    
    Tweet no. 010
    =============
    Generating with seed:
     מתאר גם כישוף אלג'יראי עם קוסקוס כואב ל
    ________________________________________
     מתאר גם כישוף אלג'יראי עם קוסקוס כואב להיה למאות בכל האותי המוצא מאירופא להוא של המופלא של היום למשמה לא משמואל (מסעות בנימין משוני מאותי ה
    
    

## Part VII: Evaluating the Model

In order to evaluate our model, we will measure the cosine distance of the tweets we generated, and the sequences in the corpus. To do that, we use the TfidfVectorizer from scikit-learn. We expect to have a distance which is not too long and not too short. If the distance is too long, then our sequences are (almost) completely random. If the distance is too short, the network overfitted the training examples and we just get the same tweets.

The `pairwise_distances` function from scikit-learn return a 2D matrix where $a_{ij}$ is the distance between sequence $i$, the generated tweet, and sequence $j$, the original.


```python
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(sequences)
Xval = vectorizer.transform(tweets)
print(pairwise_distances(Xval, Y=tfidf, metric='cosine').min(axis=1).mean())
```

    0.424759556395
    

As we can see, our result are exactly what we wanted - somewhere in the middle.
