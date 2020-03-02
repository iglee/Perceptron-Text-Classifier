# models.py

import time
import random
import math
import torch
from torch import nn
import embeddings as E
from sentiment_data import *
from collections import Counter, defaultdict
from nltk import WordNetLemmatizer, SnowballStemmer
from torch.nn import functional as F

class FeatureExtractor:

    def extract_features(self, ex_words: List[str]) -> List[int]:
        raise NotImplementedError()

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self):
        self.stem = SnowballStemmer("english")

    def extract_features(self, ex_words):
        """
        Q1: Implement the unigram feature extractor.
        Hint: You may want to use the Counter class.
        """
        sentence = [word.lower() for word in ex_words]
        sentence = [self.stem.stem(word) for word in sentence]
        return Counter(sentence)


        #raise NotImplementedError('Your code here')


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self):
        self.stem = SnowballStemmer("english")


    def extract_features(self, ex_words):
        """
        Q1: Implement the better feature extractor.
        """
        sentence = [word.lower() for word in ex_words]
        sentence = [self.stem.stem(word) for word in sentence]
        bigrams = [(sentence[i-1], sentence[i]) for i in range(1,len(sentence))]

        return Counter(sentence) + Counter(bigrams)



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
        super(PerceptronClassifier, self).__init__()
        self.feat_extractor = feat_extractor
        # parameters:
        self.corpus = []
        self.word_count = Counter()
        self.vocab = set()
        self.weight = Counter()

    def featurize(self, ex):
        """
        Converts an example into features.
        """
        return self.feat_extractor.extract_features(ex.words)

    def forward(self, feat) -> float:
        output = 0
        for x in feat:
            # check if this word is in the vocab
            if x in self.vocab:
                if x not in self.weight:
                    # if the model hasn't seen this word, initialize randomly
                    self.weight[x] = random.random()
                output += feat[x] * self.weight.get(x)
        return 1/ (1 + math.exp(-output))

    def extract_pred(self, output) -> int:
        # compute the prediction of the perceptron given the activation
        if output >= 0.5:
            return 1
        else:
            return 0

    def update_parameters(self, output, feat, ex, lr):
        # update the weight of the perceptron given its activation, the input features, the example, and the learning rate
        # check if the vocab needs updating, i.e. the model hasn't seen this sentence before
        if ex.words not in self.corpus:
            self.corpus.append(ex.words)
            self.word_count.update(feat)

            for word, count in self.word_count.items():
                if count > 2 and count < 4500:
                    self.vocab.add(word)

        # calculate loss times learning rate
        loss = (ex.label - output) * lr

        # update weight : w(t+1) = w(t) + loss * lr * input
        for x in feat:
            self.weight[x] += loss * feat[x]


class FNNClassifier(SentimentClassifier, nn.Module):
    """
    Q4: Implement the multi-layer perceptron classifier.
    """

    def __init__(self, args):
        super().__init__()
        self.glove = E.GloveEmbedding('wikipedia_gigaword', 300, default='zero')
        ### Start of your code
        self.input_dim = 300
        self.hidden_dim = 100
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim) # 1st fully connected layer
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) # 2nd fully connected layer
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
        feat = feat.sum(1)
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

        #specify BCE loss
        criterion = nn.BCELoss()

        # loss
        loss = criterion(output, target)

        # step back
        loss.backward()
        self.optim.step()

        # clear grad
        self.optim.zero_grad()
        #raise NotImplementedError('Your code here')


class RNNClassifier(FNNClassifier):

    """
    Q5: Implement the RNN classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code

        # Specify all relevant dimensions
        self.input_dim = 300
        self.hidden_dim = 20
        self.output_dim = 1
        self.max_pool_dim = 40
        self.n_layers = 1

        # specify LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)

        # layer to feed "max pooled output"
        self.linear = nn.Linear(self.max_pool_dim, self.output_dim)

        # final sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)

    def forward(self, feat):
        feat = feat.unsqueeze(0)

        # feed to LSTM (input vector is of dim : 1 x [WORD LENGTH] x EMBED DIMENSION)
        h_lstm, _ = self.lstm(feat)

        # maximum activation, with dimension squeezed, so in our case, (1 x HIDDEN*2)
        max_pool, _ = torch.max(h_lstm, 1)

        # feed to linear layer to feed to final activation [HIDDEN*2 x 1]
        output = self.linear(max_pool)

        # return sigmoid activation [1 x 1]
        return self.sigmoid(output)


class MyNNClassifier(FNNClassifier):

    """
    Q6: Implement the your own classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # Start of your code

        # Specify all relevant dimensions
        self.input_dim = 300
        self.hidden_dim = 20
        self.output_dim = 1
        self.n_layers = 1

        # specify LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)

        self.tanh = nn.Tanh()

        self.conv_layer = nn.Linear(2*self.hidden_dim + self.input_dim, self.hidden_dim)

        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        # final sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # End of your code
        self.optim = torch.optim.Adam(self.parameters(), args.learning_rate)


    def forward(self, feat):
        feat = feat.unsqueeze(0)

        # feed to LSTM (input vector is of dim : 1 x [WORD LENGTH] x EMBED DIMENSION)
        h_lstm, _ = self.lstm(feat)

        # concatenate with raw input embeddings [i.e. embedding for before and after with hidden state]
        concat = torch.cat((h_lstm, feat), 2).permute(1,0,2)
        #print(concat.shape)
        y = self.conv_layer(concat)
        #print(y)
        #print(y.shape)
        #print(h_lstm.shape)

        y = self.tanh(y).permute(0,2,1)
        #print(y.shape)
        #y = torch.max(y, 1) XX -> try max_pool1d
        #print(y)


        max_out_features = F.max_pool1d(y, y.shape[2]).squeeze(2)
        #print(max_out_features.shape)
        output = self.output_layer(max_out_features)
        #print(output.shape)
        #print(self.sigmoid(output.sum()))
        #print(y.shape)

        return self.sigmoid(output.sum())
        #raise NotImplementedError()




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
