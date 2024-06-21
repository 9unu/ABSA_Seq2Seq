from kss import split_sentences
import os
import json
def create_chunks(df, chunk_num):
    chunk_size = len(df) // chunk_num
    if len(df) % chunk_num == 0:
        chunks = [df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(chunk_num)]
    else:
        chunks = [df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(chunk_num-1)]
        chunks.append(df.iloc[chunk_num * chunk_size:])
    return chunks

def preprocess_text(text, ignores):
    for ignore in ignores:
        text = text.replace(ignore, f"|||{ignore}|||")
    return text

def postprocess_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.replace("|||", "")
        processed_sentences.append(sentence)
    return processed_sentences

def custom_split_sentences(text, ignores):
    preprocessed_text = preprocess_text(text, ignores)
    sentences = split_sentences(preprocessed_text, backend="auto")
    filtered_sentences = postprocess_sentences(sentences)
    # print(filtered_sentences)
    return filtered_sentences

def df_multiprocess(chunk):
    print(len(chunk))
    chunk['content'] = chunk.apply(lambda row: custom_split_sentences(row['content'], ignores=row['tag_text_list']), axis=1)
    print("학습용 문장분리완료")
    return chunk


def test_df_multiprocess(chunk):
    print(len(chunk))
    chunk['content'] = chunk.apply(lambda row: split_sentences(row['content']), axis=1)
    print("테스트용 문장분리완료")
    return chunk

def load_json_files(folder_path):
    file_list = os.listdir(folder_path)
    all_reviews = []
    for file_name in file_list:
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
                all_reviews.extend(data)
    return all_reviews