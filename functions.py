from kss import split_sentences
import re
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


def remove_ignored_sentences(text, ignores):
    positions = []
    for ignore in ignores:
        start = text.find(ignore)
        end = start + len(ignore)
        positions.append((start, end, ignore))
        text = text.replace(ignore, "")
    return text, positions

def insert_ignored_sentences(sentences, positions):
    for start, end, ignore in sorted(positions, reverse=True):
        for i, sentence in enumerate(sentences):
            if start <= len(sentence):
                sentences.insert(i, ignore)
                break
            start -= len(sentence) + 1
    return sentences

def custom_split_sentences(text, ignores):
    # 우선 원본 문장에서 태깅텍스트를 지움
    cleaned_text, positions = remove_ignored_sentences(text, ignores)
    
    # 태깅텍스트 지운 문장을 분리함
    sentences = split_sentences(cleaned_text)
    
    # 태깅텍스트를 지웠던 부분을 찾아서 insert함
    final_sentences = insert_ignored_sentences(sentences, positions)
    
    return final_sentences

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



def regexp(sentence):
    pattern1 = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+")  # 한글 자모음만 반복되면 공백으로 대체
    pattern2 = re.compile(r":\)|[@#$^\*\(\)\[\]\{\}<>\/\"'=+\|_]+")  # 특수문자 공백으로 대체 (~, !, %, &, -, ,, ., ;, :, ?는 유지)
    # 특수문자 공백으로 대체 (~, !, %, &, -, ,, ., ;, :, ?는 유지)
    # pattern3 = re.compile(r"([^\d])\1{2,}")  # 숫자를 제외한 동일한 문자 3개 이상이면 공백으로 대체
    pattern3 = re.compile(  # 이모티콘 공백으로 대체
        "["                               
        "\U0001F600-\U0001F64F"  # 감정 관련 이모티콘
        "\U0001F300-\U0001F5FF"  # 기호 및 픽토그램
        "\U0001F680-\U0001F6FF"  # 교통 및 지도 기호
        "\U0001F1E0-\U0001F1FF"  # 국기 이모티콘
        "]+", flags=re.UNICODE
    )
    new_sent1 = pattern1.sub(' ', sentence)
    new_sent2 = pattern2.sub(' ', new_sent1)
    new_sent3 = pattern3.sub(' ', new_sent2)
    # new_sent4 = pattern4.sub(' ', new_sent3)
    return new_sent3
