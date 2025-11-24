import json
import re
from collections import Counter
import jieba

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name

        
        if self.dataset_name == 'PED_xray':
            self.clean_report = self.clean_report_PED_xray
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        self.ann = json.loads(open(self.ann_path,  'r', encoding="utf_8_sig").read())

        self.dict_pth = args.dict_pth
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        if self.dict_pth != '':
            word_dict = json.loads(open(self.dict_pth, 'r', encoding="utf_8_sig").read())
            return word_dict[0], word_dict[1]
        else:
            total_tokens = []
            split_list = ['train', 'test', 'val']
            for split in split_list:
                for example in self.ann[split]:
                    tokens = list(jieba.lcut(example['report']))
                    for token in tokens:
                        total_tokens.append(token)
            counter = Counter(total_tokens)
            vocab = [k for k, v in counter.items()] + ['<unk>']
            token2idx, idx2token = {}, {}
            for idx, token in enumerate(vocab):
                token2idx[token] = idx + 1
                idx2token[idx + 1] = token
            return token2idx, idx2token

    def clean_report_PED_xray(self, report):
        # 1. 替换无效字符
        report_cleaner = lambda t: t.replace('\n', '') \
            .replace('\r', '') \
            .replace('、', '，') \
            .replace('。', '。') \
            .replace('．', '。') \
            .replace('  ', '') \
            .strip()

        # 2. 分句：使用中文句号 '。' 拆句（也可以加上 '；'、'！' 等）
        sentence_splitter = lambda t: re.split(r'[。！？]', t)

        # 3. 分词：每句用 jieba.lcut 进行分词
        sent_cleaner = lambda t: ' '.join(jieba.lcut(t.strip()))

        # 4. 清洗 + 分词
        cleaned_sentences = []
        for sent in sentence_splitter(report_cleaner(report)):
            if sent.strip() != '':
                cleaned_sentences.append(sent_cleaner(sent))

        # 5. 拼接成字符串（加上句子间的 ' . '）
        report = ' . '.join(cleaned_sentences) + ' .'
        return report


    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_list(self, ids):
        txt = []
        for i, idx in enumerate(ids):
            if idx > 0:
                txt.append(self.idx2token[idx])
            else:txt.append('<start/end>')

        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

    def decode_batch_list(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode_list(ids))
        return out

