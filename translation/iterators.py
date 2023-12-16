
from transformers import PreTrainedTokenizerFast



def collate_batch_huggingface(
    batch, 
    src_language: str,
    tgt_language: str,
    src_tokenizer: PreTrainedTokenizerFast, 
    tgt_tokenizer: PreTrainedTokenizerFast,
    device, 
    max_padding=128
):

    def wrap_special(sentence):
        # Wrap sentence with BOS and EOS tokens
        return src_tokenizer.bos_token + sentence + src_tokenizer.eos_token
    
    src_texts = [wrap_special(item['translation'][src_language]) for item in batch]
    tgt_texts = [wrap_special(item['translation'][tgt_language]) for item in batch]

    # Tokenize and pad source sequences
    src_encodings = src_tokenizer(src_texts, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=max_padding)
    src_ids = src_encodings['input_ids'].to(device)

    # Tokenize and pad target sequences
    tgt_encodings = tgt_tokenizer(tgt_texts, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=max_padding)
    tgt_ids = tgt_encodings['input_ids'].to(device)

    return src_ids, tgt_ids




