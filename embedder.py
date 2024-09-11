from sentence_transformers import SentenceTransformer
import nlpaug.augmenter.word as naw
import translators as ts
import pandas as pd
import numpy as np


def append_row(df, row):
    return pd.concat([
                df, 
                pd.DataFrame([row], columns=row.index)]
           ).reset_index(drop=True)


data_file = '/home/mkr/lama3/data/train.csv'
output_file = '/home/mkr/lama3/data/train.json'
translator = 'yandex'


input_data = pd.read_csv(data_file, keep_default_na=False)
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True)

aug_w2v = naw.WordEmbsAug(
    #model_type='word2vec', model_path='../input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin',
    model_type='glove', model_path='/home/mkr/lama3/data/glove.6B.100d.txt',
    action="substitute")

# Create the pandas DataFrame
output_data = pd.DataFrame(columns=['ID', 'Label', 'KeyWord', 'Sentence'])

_ = ts.preaccelerate_and_speedtest()

for idx in range(input_data.shape[0]):
    try:
        id = input_data.iloc[idx, 0]
        label = input_data.iloc[idx, 4]
        
        #add source data
        key_word = model.encode(input_data.iloc[idx, 1])
        sentence = model.encode(input_data.iloc[idx, 3])
        new_row = pd.Series({'ID': id, 'Label': label, 'KeyWord': key_word, 'Sentence':  sentence})
        output_data = append_row(output_data, new_row)
        
        #add backtranslated data
        fr_text = ts.translate_text(input_data.iloc[idx, 3], translator=translator, from_language='en', to_language='fr')
        sentence = model.encode(ts.translate_text(fr_text, translator=translator, from_language='fr', to_language='en'))
        new_row = pd.Series({'ID': id, 'Label': label, 'KeyWord': key_word, 'Sentence':  sentence})
        output_data = append_row(output_data, new_row)

        #add 2nd backtranslated data
        ru_text = ts.translate_text(input_data.iloc[idx, 3], translator=translator, from_language='en', to_language='ru')
        de_text = ts.translate_text(ru_text, translator=translator, from_language='ru', to_language='de')
        sentence = model.encode(ts.translate_text(de_text, translator=translator, from_language='de', to_language='en'))
        new_row = pd.Series({'ID': id, 'Label': label, 'KeyWord': key_word, 'Sentence':  sentence})
        output_data = append_row(output_data, new_row)
        
        #add agumented data
        agumented_text = aug_w2v.augment(input_data.iloc[idx, 3])
        sentence = model.encode(agumented_text[0])
        new_row = pd.Series({'ID': id, 'Label': label, 'KeyWord': key_word, 'Sentence':  sentence})
        output_data = append_row(output_data, new_row)
        
        #add backtranslated agumented data
        fr_text = ts.translate_text(agumented_text[0], translator=translator, from_language='en', to_language='fr')
        sentence = model.encode(ts.translate_text(fr_text, translator=translator, from_language='fr', to_language='en'))
        new_row = pd.Series({'ID': id, 'Label': label, 'KeyWord': key_word, 'Sentence':  sentence})
        output_data = append_row(output_data, new_row)
    
    except:
        print("Something else went wrong")
 
output_data.to_json(output_file, orient = 'split') 