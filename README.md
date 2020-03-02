## Perceptron Sentiment Classification of Text

### 1. Single Perceptron Sentiment Classification
- Pre-processing: lowercasing, frequency of word count to limit vocabulary, NLTK Snowball stemming
- Unigram Feature
- Unigram and Bigram Features


### 2. Multilayer Perceptron Sentiment Classification
- two layer with tanh activation layer, with sigmoid activation for final output (used BCE loss)

### 3. Recurrent Neural Networks
- Bi-directional LSTM with "max pooling" activation to a linear fully connected layer.  Then, this is fed to final Sigmoid activation.

### 4. RCNN
- Bi-directional LSTM with convolution and max pooling.
