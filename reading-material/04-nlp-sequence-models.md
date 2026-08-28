# Model 4 — NLP & Sequence Models: Reading Material

Covers: text preprocessing, TF-IDF, word embeddings, RNNs, attention basics.

## Core NLP fundamentals
- **Hugging Face NLP Course** (free, official, very hands-on):
  https://huggingface.co/learn/nlp-course
- **scikit-learn — Working with text data** (official tutorial, covers
  bag-of-words and TF-IDF with real code):
  https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

## Word embeddings
- **Jay Alammar — The Illustrated Word2vec**: the best visual explanation of
  where word embeddings come from and why they capture meaning.
  https://jalammar.github.io/illustrated-word2vec/

## RNNs
- **Christopher Olah — Understanding LSTM Networks**: the canonical
  explanation of why vanilla RNNs struggle with long sequences and how
  LSTMs' gating fixes it. Directly relevant to this model's vanishing
  gradient exercise. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **PyTorch — NLP From Scratch tutorial series** (official docs): builds an
  RNN for name classification/generation, matches this model's style closely.
  https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

## Attention & Transformers (why they replaced RNNs for most tasks)
- **Jay Alammar — The Illustrated Transformer**: the most-cited visual
  explanation of self-attention and transformer architecture that exists.
  https://jalammar.github.io/illustrated-transformer/
- **Hugging Face — How do Transformers work?** (part of the NLP course
  above, shorter and more applied): https://huggingface.co/learn/nlp-course/chapter1/4
