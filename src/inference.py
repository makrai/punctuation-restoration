import re
import torch

import argparse
from model import DeepPunctuation, DeepPunctuationCRF
from config import *

parser = argparse.ArgumentParser(description='Punctuation restoration inference on text file')
parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
parser.add_argument('--pretrained-model', default='xlm-roberta-large', type=str, help='pretrained language model')
parser.add_argument('--lstm-dim', default=-1, type=int,
                    help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use CRF layer or not')
parser.add_argument('--language', default='en', type=str, help='language English (en) oe Bangla (bn)')
parser.add_argument('--in-file', default='data/test_en.txt', type=str, help='path to inference file')
parser.add_argument('--weight-path', default='xlm-roberta-large.pt', type=str, help='model weight path')
parser.add_argument('--sequence-length', default=256, type=int,
                    help='sequence length to use when preparing dataset (default 256)')
parser.add_argument('--out-file', default='data/test_en_out.txt', type=str, help='output file location')

PATH = '/mnt/permanent/Language/English/LangMod/punctuation-restoration/'

class Inferencer():
    def __init__(self, cuda=True, pretrained_model='roberta-large',
            lstm_dim=-1, use_crf=False, language='en',
            weight_path=PATH+'roberta-large-en.pt', sequence_length=256):
        self.use_crf=use_crf
        self.language=language
        self.sequence_length=sequence_length

        # tokenizer
        self.tokenizer = MODELS[pretrained_model][1].from_pretrained(pretrained_model)
        self.token_style = MODELS[pretrained_model][3]

        # logs
        model_save_path = weight_path

        # Model
        self.device = torch.device('cuda' if (cuda and torch.cuda.is_available()) else 'cpu')
        if self.use_crf:
            self.deep_punctuation = DeepPunctuationCRF(pretrained_model, freeze_bert=False, lstm_dim=lstm_dim)
        else:
            self.deep_punctuation = DeepPunctuation(pretrained_model, freeze_bert=False, lstm_dim=lstm_dim)
        self.deep_punctuation.to(self.device)
        self.deep_punctuation.load_state_dict(torch.load(model_save_path))
        self.deep_punctuation.eval()

    def inference(self, text):
        text = re.sub(r"[,:\-–.!;?]", '', text)
        words_original_case = text.split()
        words = text.lower().split()

        word_pos = 0
        sequence_len = self.sequence_length
        result = ""
        decode_idx = 0
        punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
        if self.language != 'en':
            punctuation_map[2] = '।'

        while word_pos < len(words):
            x = [TOKEN_IDX[self.token_style]['START_SEQ']]
            y_mask = [0]

            while len(x) < sequence_len and word_pos < len(words):
                tokens = self.tokenizer.tokenize(words[word_pos])
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(self.tokenizer.convert_tokens_to_ids(tokens[i]))
                        y_mask.append(0)
                    x.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
                    y_mask.append(1)
                    word_pos += 1
            x.append(TOKEN_IDX[self.token_style]['END_SEQ'])
            y_mask.append(0)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
            attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x]

            x = torch.tensor(x).reshape(1,-1)
            y_mask = torch.tensor(y_mask)
            attn_mask = torch.tensor(attn_mask).reshape(1,-1)
            x, attn_mask, y_mask = x.to(self.device), attn_mask.to(self.device), y_mask.to(self.device)

            with torch.no_grad():
                if self.use_crf:
                    y = torch.zeros(x.shape[0])
                    y_predict = self.deep_punctuation(x, attn_mask, y)
                    y_predict = y_predict.view(-1)
                else:
                    y_predict = self.deep_punctuation(x, attn_mask)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
            for i in range(y_mask.shape[0]):
                if y_mask[i] == 1:
                    result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                    decode_idx += 1
        return result


if __name__ == '__main__':
    args = parser.parse_args()
    inferencer = Inferencer(args.cuda, args.pretrained_model, args.lstm_dim,
        args.use_crf, args.language, args.in_file, args.weight_path,
        args.sequence_length)
    with open(args.in_file, 'r', encoding='utf-8') as f:
        text = f.read()
    inferencer.inference(text)
