import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import archive_model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.training.trainer import Trainer

from classifier.sst_classifier import LstmClassifier
from classifier.transformer_encoder import TransformerSeq2VecEncoder

# Constants
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
MAX_SEQ_LEN = 100
SERIALIZATION_DIR = './serializaed-bert-transformer'

token_indexer = PretrainedBertIndexer(pretrained_model='bert-base-uncased',
                                      max_pieces=MAX_SEQ_LEN,
                                      do_lowercase=True)


def tokenizer(s: str):
    return token_indexer.wordpiece_tokenizer(s)[:MAX_SEQ_LEN - 2]


reader = StanfordSentimentTreeBankDatasetReader(token_indexers={'tokens': token_indexer})

train_dataset = reader.read('data/stanfordSentimentTreebank/trees/train.txt')
dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')

# You can optionally specify the minimum count of tokens/labels.
# `min_count={'tokens':3}` here means that any tokens that appear less than three times
# will be ignored and not included in the vocabulary.
vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                  min_count={'tokens': 3})

# token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
#                             embedding_dim=EMBEDDING_DIM)

# BasicTextFieldEmbedder takes a dict - we need an embedding just for tokens,
# not for labels, which are used as-is as the "answer" of the sentence classification
# word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

bert_embedder = PretrainedBertEmbedder(pretrained_model='bert-base-uncased', top_layer_only=True)

word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                            allow_unmatched_keys=True)

# Seq2VecEncoder is a neural network abstraction that takes a sequence of something
# (usually a sequence of embedded word vectors), processes it, and returns a single
# vector. Oftentimes this is an RNN-based architecture (e.g., LSTM or GRU), but
# AllenNLP also supports CNNs and other simple architectures (for example,
# just averaging over the input vectors).
# encoder = PytorchSeq2VecWrapper(
#     torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

# HW
encoder = TransformerSeq2VecEncoder(EMBEDDING_DIM,
                                    HIDDEN_DIM,
                                    projection_dim=128,
                                    feedforward_hidden_dim=128,
                                    num_layers=2,
                                    num_attention_heads=2)

model = LstmClassifier(word_embeddings, encoder, vocab)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

iterator = BucketIterator(batch_size=64, sorting_keys=[("tokens", "num_tokens")])

iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=dev_dataset,
                  cuda_device=0,
                  patience=5,
                  num_epochs=20)

metrics = trainer.train()

print(metrics)

