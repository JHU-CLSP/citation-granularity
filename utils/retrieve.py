import re
from nltk.tokenize import PunktSentenceTokenizer

def text_split_by_punctuation(original_text, return_dict=False):
    text = original_text
    custom_sent_tokenizer = PunktSentenceTokenizer()
    punctuations = r"([。；！？])"  # For Chinese support

    separated = custom_sent_tokenizer.tokenize(text)
    separated = sum([re.split(punctuations, s) for s in separated], [])
    # Put the punctuations back to the sentence
    for i in range(1, len(separated)):
        if re.match(punctuations, separated[i]):
            separated[i-1] += separated[i]
            separated[i] = ''

    separated = [s for s in separated if s != ""]
    if len(separated) == 1:
        separated = original_text.split('\n\n')
    separated = [s.strip() for s in separated if s.strip() != ""]
    
    if not return_dict:
        return separated
    else:
        pos = 0
        res = []
        for i, sent in enumerate(separated):
            st = original_text.find(sent, pos)
            assert st != -1, sent
            ed = st + len(sent)
            res.append(
                {
                    'c_idx': i,
                    'content': sent,
                    'start_idx': st,
                    'end_idx': ed,
                }
            )
            pos = ed
        return res

def text_split_by_n_sentences(original_text, sentences_per_chunk=5, return_dict=False):
    """Split text into chunks of N sentences"""
    # First get all sentences
    sentences = text_split_by_punctuation(original_text, return_dict=True)
    
    if not return_dict:
        # Simple list of text chunks
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = ' '.join([s['content'] for s in chunk_sentences])
            chunks.append(chunk_text)
        return chunks
    else:
        # Detailed dict with position info
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            if chunk_sentences:
                chunk_text = ' '.join([s['content'] for s in chunk_sentences])
                chunks.append({
                    'c_idx': len(chunks),
                    'content': chunk_text,
                    'start_idx': chunk_sentences[0]['start_idx'],
                    'end_idx': chunk_sentences[-1]['end_idx'],
                    'sentence_indices': list(range(i, min(i + sentences_per_chunk, len(sentences))))
                })
        return chunks
