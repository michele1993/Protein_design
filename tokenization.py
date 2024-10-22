import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence


class Seq_Label_Tokenization():
    """ 
    Class to perform all pre-processing operation of Tokenization, Numericalization, adding special tokens and convert to Tensor
    for Dataset returning tuple of seq and label.
    """

    def __init__(self, clean_dataset, aminoacid_data=False):
        """ Args:
                clean_dataset = dataset of already cleaned data in format sequence and labels
                aminoacid_data = whether the data involves aminoacid sequences
        """

        # Initialise tokenizer function
        self.tokenizer = get_tokenizer(self.char_tokenizer)

        # Define special tokens: UNK: unknown token, PAD: padding token, BOS: beginning (of setence) token, EOS: end (of setence) token
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3 
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

        # Define  Numericalization function (vocab)
        self.vocab = build_vocab_from_iterator(self._yield_tokens(clean_dataset), min_freq=1, specials=special_symbols, special_first=True)

        # Precution: set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
        # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
        self.vocab.set_default_index(self.UNK_IDX)

        # We know how many unique index token we should have if we are using aminoacid data, so we can check the solution
        if aminoacid_data:
            n_aminoAcid_letters = 20
            assert len(self.vocab.get_stoi()) == len(special_symbols) + n_aminoAcid_letters, "Wrong n. of unique index tokens, check numericalization process!"

    def collate_fn(self, batch):
        """ 
        Collate function to perform all operations on a data batch.
        This function can be passed to DataLoader to prepare each batch appropriately.
        """

        seq_batch, label_batch = [], []

        for seq, label in batch:
            tokens = self.tokenizer(seq) # Tokenization
            index = self.vocab(tokens)   # Numericalization
            tensor_seq = self._tensor_transform(index) # add BOS/EOS and transform to Tensor
            # Append transformed sequence
            seq_batch.append(tensor_seq)
            label_batch.append(label)

        # Add padding: to  store in batch tensor all seq must have same length within the batch
        padded_seq_batch = pad_sequence(seq_batch, padding_value=self.PAD_IDX)

        return padded_seq_batch.T, torch.tensor(label_batch)

    def char_tokenizer(self, text):    
        """ Method for character tokenizer"""
        return list(text)

    def _yield_tokens(self, data):
        """ Helper method to create iterable for build_vocab_from_iterator """
        for seq_sample, _ in data:
            yield self.tokenizer(seq_sample)

    def _tensor_transform(self, token_ids):
        """ Add BOS and EOS respectively at beginning and end of seq and transform to tensor
        Args:
            token_ids: list of token idxs
        """
        return torch.cat([torch.tensor([self.BOS_IDX]), torch.tensor(token_ids), torch.tensor([self.EOS_IDX])])
