import os
import pickle
import urllib.request
import string
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

def clean_caption(caption):
    # Same cleaning as in the notebook
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = " ".join([word for word in caption.split() if len(word) > 1 and word.isalpha()])
    return caption

def main():
    print("Downloading Flickr8k.token.txt...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/Flickr8k_text/Flickr8k.token.txt"
    token_file = "Flickr8k.token.txt"
    if not os.path.exists(token_file):
        urllib.request.urlretrieve(url, token_file)
    
    print("Parsing captions...")
    captions_by_image = {}
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) < 2:
                continue
            # format: image_id#0   caption
            image_id = tokens[0].split('#')[0]
            caption = tokens[1]
            
            image_id = image_id.split('.')[0]
            caption = clean_caption(caption)
            caption = "startseq " + caption + " endseq"
            
            if image_id not in captions_by_image:
                captions_by_image[image_id] = []
            captions_by_image[image_id].append(caption)

    all_captions_list = [caption for captions in captions_by_image.values() for caption in captions]
    
    print(f"Total captions: {len(all_captions_list)}")
    
    caption_tokenizer = Tokenizer()
    caption_tokenizer.fit_on_texts(all_captions_list)
    vocabulary_size = len(caption_tokenizer.word_index) + 1
    max_caption_length = max(len(caption.split()) for caption in all_captions_list)
    
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Max caption len: {max_caption_length}")
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    tokenizer_path = os.path.join('models', 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(caption_tokenizer, f)
        
    print(f"Tokenizer saved to {tokenizer_path} ✅")

if __name__ == "__main__":
    main()
