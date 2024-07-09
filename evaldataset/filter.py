import json
import nlqf
import torch
def get_rule_filter(raw_comments):
    import nlqf
    rule_list = ['contain_html_tag', 'detach_brackets', 'detach_html_tag'
        , 'end_with_question_mark', 'less_than_three_words', 'contain_any_non_English'
        , 'contain_any_url', 'contain_any_javadoc_tag', 'not_contain_any_letter']

    def my_rule1(comment_str):
        return 5 < len(comment_str) < 15

    def my_rule2(comment_str):
        if comment_str.startswith('Return'):
            return comment_str.replace('Return', '')
        else:
            return True

    my_rule_dict = {
        'length_less_than_10': my_rule1,
        'detach_return': my_rule2
    }

    comments, idx = nlqf.rule_filter(raw_comments, \
                                     selected_rules=rule_list, defined_rule_dict=my_rule_dict)

    return comments, idx


def get_model_filter(raw_comments):
    
    vocab_path = 'VAE/word_vocab.json'
    model_path = 'VAE/vae.model'
    with open(vocab_path, 'r') as f:
        word_vocab = json.load(f)
    model = torch.load(model_path)

    queries, idx = nlqf.model_filter(raw_comments, word_vocab, model,
                                     with_cuda=True, query_len=20, num_workers=10, max_iter=1000)

    return queries, idx

from tqdm import tqdm

with open("qq2",'r') as f:
    raw_comments = f.read().split('\n')
with open("code3.txt",'r') as f:
    code = f.read().split('\n')


_, idx = get_rule_filter(raw_comments)

raw_comments = [raw_comments[i] for i in idx]
code = [code[i] for i in idx]

with open("desc_filter2.txt",'w') as f:
    f.write('\n'.join(raw_comments))


_, idx = get_model_filter(raw_comments)
raw_comments = [raw_comments[i] for i in idx]
code = [code[i] for i in idx]

with open("desc_filter2.txt",'w') as f:
    f.write('\n'.join(raw_comments))

with open("code_filter2.txt",'w') as f:
    f.write('\n'.join(code))