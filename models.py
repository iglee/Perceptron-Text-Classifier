# models.py

import time
import random
import math
import torch
from torch import nn
import embeddings as E
from sentiment_data import *
from collections import Counter, defaultdict


################################
# build vocabulary for UNIGRAM #
################################
train = read_sentiment_examples("data/train.txt")
c = Counter()
for sentence in train:
    for word in sentence.words:
        c.update([word.lower()])

vocab = Counter()

for x in c:
    if c[x] > 3 and c[x] < 800:
        vocab[x] = c[x]
    else:
        vocab["<UNK>"] += c[x]

word_to_ix = {}

for word in vocab.keys():
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)



###############################
# build vocabulary for BETTER #
###############################

c_bigrams = Counter()
for sentence in train:
    for i in range(1,len(sentence.words)):
        c_bigrams.update([(sentence.words[i-1].lower(),sentence.words[i].lower())])

vocab_bigrams = Counter()
for x in c_bigrams:
    if c_bigrams[x] > 3 and c_bigrams[x] < 500:
        vocab_bigrams[x] = c_bigrams[x]
    else:
        vocab_bigrams["<UNK>"] += c_bigrams[x]

word_to_ix_bigrams = {"<LEN>" : 0} # length of sentence

for word in vocab_bigrams.keys():
    if word not in word_to_ix_bigrams:
        word_to_ix_bigrams[word] = len(word_to_ix_bigrams)

BIGRAM_VOCAB_SIZE = len(word_to_ix_bigrams)


class FeatureExtractor:

    def extract_features(self, ex_words: List[str]) -> List[int]:
        raise NotImplementedError()

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def extract_features(self, ex_words):
        """
        Q1: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """
        vec = torch.zeros((1, len(word_to_ix)))

        words = []
        for word in ex_words:
            words.append(word.lower())

        c = Counter(words)

        for word in c:
            x = word.lower()
            if x in word_to_ix:
                vec[0][word_to_ix[x]] = c[word]
            else:
                vec[0][word_to_ix["<UNK>"]] += c[word]

        return vec.to_sparse()


        #raise NotImplementedError('Your code here')


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def extract_features(self, ex_words):
        """
        Q3: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """
        vec = torch.zeros((1, len(word_to_ix_bigrams)))

        bigrams = []
        for i in range(1, len(ex_words)):
            bigrams.append((ex_words[i-1].lower(),ex_words[i].lower()))

        bigram_count_sent = Counter(bigrams)

        for bigram in bigram_count_sent:
            if bigram in word_to_ix_bigrams:
                vec[0][word_to_ix_bigrams[bigram]] = bigram_count_sent[bigram]
            else:
                vec[0][word_to_ix_bigrams["<UNK>"]] += bigram_count_sent[bigram]

        vec[0][0] = len(ex_words)
        return vec.to_sparse()

        # raise NotImplementedError('Your code here')


class SentimentClassifier(object):

    def featurize(self, ex):
        raise NotImplementedError()

    def forward(self, feat):
        raise NotImplementedError()

    def extract_pred(self, output):
        raise NotImplementedError()

    def update_parameters(self, output, feat, ex, lr):
        raise NotImplementedError()

    def run_train(self, train_data: List[SentimentExample], dev_data: List[SentimentExample], lr=1e-3, epoch=20):
        """
        Training loop.
        """
        train_data = train_data[:]
        for ep in range(epoch):
            start = time.time()
            random.shuffle(train_data)

            if isinstance(self, nn.Module):
                self.train()

            acc = []
            for ex in train_data:
                feat = self.featurize(ex)
                output = self.forward(feat)
                self.update_parameters(output, feat, ex, lr)
                predicted = self.extract_pred(output)
                acc.append(predicted == ex.label)
            acc = sum(acc) / len(acc)

            if isinstance(self, nn.Module):
                self.eval()

            dev_acc = []
            for ex in dev_data:
                feat = self.featurize(ex)
                output = self.forward(feat)
                predicted = self.extract_pred(output)
                dev_acc.append(predicted == ex.label)
            dev_acc = sum(dev_acc) / len(dev_acc)
            print('epoch {}: train acc = {}, dev acc = {}, time = {}'.format(ep, acc, dev_acc, time.time() - start))

    def predict(self, ex: SentimentExample) -> int:
        feat = self.featurize(ex)
        output = self.forward(feat)
        return self.extract_pred(output)


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, ex: SentimentExample) -> int:
        return 1

    def run_train(self, train_data: List[SentimentExample], dev_data: List[SentimentExample], lr=None, epoch=None):
        pass


class PerceptronClassifier(SentimentClassifier):
    """
    Q1: Implement the perceptron classifier.
    """

    def __init__(self, feat_extractor):
        self.feat_extractor = feat_extractor
        # parameters:
        if isinstance(self.feat_extractor,UnigramFeatureExtractor):
            self.inputSize = VOCAB_SIZE
            self.hiddenSize = 600
        if isinstance(self.feat_extractor,BetterFeatureExtractor):
            self.inputSize = BIGRAM_VOCAB_SIZE
            self.hiddenSize = 500


        self.outputSize = 1


        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)

        # bias
        self.b1 = torch.randn((1, self.hiddenSize))
        self.b2 = torch.randn((1, self.outputSize))
        #raise NotImplementedError('Your code here')

    def featurize(self, ex):
        """
        Converts an example into features.
        """
        return self.feat_extractor.extract_features(ex.words)

    def forward(self, feat) -> float:
        # compute the activation of the perceptron
        self.z = torch.matmul(feat, self.W1) + self.b1
        self.z2 = 1/ (1 + torch.exp(-self.z)) # sigmoid
        #self.z2 = self.z.clamp(min=0) # ReLU
        self.z3 = torch.matmul(self.z2, self.W2) + self.b2
        output = 1/ (1 + torch.exp(-self.z3)) # sigmoid
        #output = self.z3.clamp(min=0)
        return output
        #raise NotImplementedError('Your code here')

    def extract_pred(self, output) -> int:
        # compute the prediction of the perceptron given the activation
        if output >= 0.5:
            return 1
        else:
            return 0

        #raise NotImplementedError('Your code here')

    def derivative_sigmoid(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def derivative_relu(self, s):
        temp = s.clone()
        temp[s>0] = 1
        temp[s<0] = 0
        return temp

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input features, the example, and the learning rate
        self.loss = (ex.label - output) ** 2

        self.delta_out = self.derivative_sigmoid(output)
        #self.delta_out = self.derivative_relu(output)
        self.delta_hidden = self.derivative_sigmoid(self.z2)
        #self.delta_hidden = self.derivative_relu(self.z2)

        self.d_outp = 2 * (ex.label - output) * self.delta_out
        self.loss_h = torch.mm(self.d_outp, self.W2.t())
        self.d_hidn = self.loss_h * self.delta_hidden

        self.W2 += torch.mm(self.z2.t(), self.d_outp) * lr
        self.W1 += torch.mm(feat.t(), self.d_hidn) * lr

        self.b2 += self.d_outp.sum()
        self.b1 += self.d_hidn.sum()

        #raise NotImplementedError('Your code here')


class FNNClassifier(SentimentClassifier, nn.Module):
    """
    Q4: Implement the multi-layer perceptron classifier.
    """

    def __init__(self, args):
        super().__init__()
        self.glove = E.GloveEmbedding('wikipedia_gigaword', 300, default='zero')
        ### Start of your code
        self.inputSize = 300
        self.hiddenSize = 100
        self.outputSize = 1

        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize) # 1st fully connected layer
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hiddenSize, self.outputSize) # 2nd fully connected layer
        self.sigmoid = nn.Sigmoid()

        #raise NotImplementedError('Your code here')

        ### End of your code

        # do not touch this line below
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def featurize(self, ex):
        # You do not need to change this function
        # return a [T x D] tensor where each row i contains the D-dimensional embedding for the ith word out of T words
        embs = [self.glove.emb(w.lower()) for w in ex.words]
        return torch.Tensor(embs)

    def forward(self, feat) -> torch.Tensor:
        # compute the activation of the FNN
        feat = feat.unsqueeze(0)
        hidden1 = self.fc1(feat)
        tanh_activation = self.tanh(hidden1)
        hidden2 = self.fc2(tanh_activation)
        output = self.sigmoid(hidden2)
        return output
        #raise NotImplementedError('Your code here')

    def extract_pred(self, output) -> int:
        # compute the prediction of the FNN given the activation
        if output >= 0.5:
            return 1
        else:
            return 0
        #raise NotImplementedError('Your code here')

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input features, the example, and the learning rate
        target = torch.Tensor([[ex.label]])

        raise NotImplementedError('Your code here')


class RNNClassifier(FNNClassifier):

    """
    Q5: Implement the RNN classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code

        raise NotImplementedError('Your code here')

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def forward(self, feat):
        feat = feat.unsqueeze(0)
        raise NotImplementedError('Your code here')


class MyNNClassifier(FNNClassifier):

    """
    Q6: Implement the your own classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code

        raise NotImplementedError('Your code here')

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def forward(self, feat):
        feat = feat.unsqueeze(0)

        raise NotImplementedError('Your code here')


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You don't need to change this.
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor()
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor()
    else:
        raise Exception("Pass in UNIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = PerceptronClassifier(feat_extractor)
    elif args.model == "FNN":
        model = FNNClassifier(args)
    elif args.model == 'RNN':
        model = RNNClassifier(args)
    elif args.model == 'MyNN':
        model = MyNNClassifier(args)
    else:
        raise NotImplementedError()

    model.run_train(train_exs, dev_exs, lr=args.learning_rate, epoch=args.epoch)
    return model
