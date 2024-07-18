import json
import os
import pandas as pd
from functions import *
import multiprocessing as mp
import time 

def make_df(folder_path):
    '''score 없는 버전'''
    folder_path = folder_path
    df = load_json_files(folder_path)
    df = pd.json_normalize(df)
    # 줄바꿈 문자 제거
    df['content'] = df['content'].apply(lambda row: row.replace("\n", "")).apply(lambda row: regexp(row))
    
    # 태깅정보가 없는 애들은 테스트용으로 따로 저장
    not_tagged = df[df['our_topics'].isnull()].index
    test_df = df.loc[not_tagged]
    
    df.dropna(subset=['our_topics'], inplace=True)
    
    # 태그 정보를 1차원 리스트로 변환
    df['tag_text_list'] = None
    df['tag_topic_list'] = None

    total_df = pd.DataFrame()
    '''태그 정보 유무'''
    tag_exist = False
    
    for index, row in df.iterrows():
        tagged_text_list = []
        tagged_topic_list = []

        for tag in row['our_topics']:
            try:
                if ("topic" not in tag
                        or "positive_yn" not in tag):
                    continue
                else:
                    # print(regexp(tag['text'].replace("\n", ' ')))
                    tagged_text_list.append(regexp(tag['text'].replace("\n", ' ')))
                    tagged_topic_list.append(tag['topic'])
                        
            except:
                continue  # 오류 발생 시 다음 태그로 넘어감
        

        if tagged_text_list != [] and tagged_topic_list != []:
            tag_exist = True
            df.at[index, 'tag_text_list'] = tagged_text_list
            df.at[index, 'tag_topic_list'] = tagged_topic_list
    

    if tag_exist:
        '''테스트용 텍스트 문장 분리 멀티 프로세스'''
        not_tagged = df[df['tag_text_list'].isnull()].index     # our_topic은 있지만, 태깅은안되어잇는애들도 테스트용으로 추가
        test_df = pd.concat([test_df, df.loc[not_tagged]])
        print("테스트 텍스트 수:", len(test_df))
        chunk_num = mp.cpu_count()
        if len(test_df) > chunk_num:
            chunks = create_chunks(test_df, chunk_num)
            print("분산 청크 수:", len(chunks))
            pool = mp.Pool(processes=chunk_num)
            dfs = pool.map(test_df_multiprocess, chunks)
            pool.close()
            pool.join()
            test_df = pd.concat(dfs)
            test_text_list=[]
            print(test_df.head())
            for text_list in test_df['content']:
                for text in text_list:
                    test_text_list.append(text)
            test_df=pd.DataFrame({'sentence':test_text_list})
        else:
            test_df['content'] = test_df.apply(lambda row: split_sentences(row['content'], ignores=row['tag_text_list']), axis=1)
            test_text_list=[]
            for text in test_df['content']:
                for text in text_list:
                    test_text_list.append(text)
            test_df=pd.DataFrame({'sentence':test_text_list})
        df.dropna(subset=['tag_text_list'], inplace=True)

        '''학습용 문장 분리 멀티 프로세싱'''
        print("학습용 데이터 수:", len(df))
        # mp.freeze_support()
        chunk_num = mp.cpu_count()
        if len(df) > chunk_num:
            chunks = create_chunks(df, chunk_num)
            print("분산 청크 수:", len(chunks))
            pool = mp.Pool(processes=chunk_num)
            dfs = pool.map(df_multiprocess, chunks)
            pool.close()
            pool.join()
            df = pd.concat(dfs)
        else:
            df['content'] = df.apply(lambda row: custom_split_sentences(row['content'], ignores=row['tag_text_list']), axis=1)

        # 분리된 문장에 태깅한 문장이 있으면 dict로 매칭하기
        origin_sentence = []
        total_tag_list = []
        for index, row in df.iterrows():
            for origin_text in row['content']:  # 리스트의 각 요소를 순회
                tag_list = []
                origin_sentence.append(origin_text)
            
                for tag_text, tag_topic in zip(
                        row['tag_text_list'],
                        row['tag_topic_list']):
                        
                    # 태그된 문장이 속한 문장이면
                    if tag_text in origin_text:
                        tag_dict = {
                                'text': tag_text,
                                'topic': tag_topic,
                        }
                        tag_list.append(tag_dict)
                    
                if not tag_list:
                    tag_list.append({'text': "태그 없음"})
                    
                total_tag_list.append(tag_list)

        sentenced_df = pd.DataFrame({'sentence': origin_sentence, 'tag_list': total_tag_list})

        # 앞에서 매칭한 태그 정보를 기반으로 model의 output으로 받을 텍스트 생성
        sentenced_df['tag_info'] = ""
        for index, row in sentenced_df.iterrows():
            total_str = ""
            for tag in row['tag_list']:
                text = tag['text']
                if text == "태그 없음":
                    total_str = '태그 없음'
                    break
                topic = tag['topic']

                total_str += f"{text} : ({topic}) // "
            sentenced_df.at[index, 'tag_info'] = total_str.rstrip(" // ")

        # total_df에 데이터프레임 추가
        total_df = pd.concat([total_df, sentenced_df], ignore_index=True)
                
    return total_df[['sentence', 'tag_info']], test_df['sentence']

if __name__ == '__main__':
    '''스코어 포함 case'''
    df , not_tagged_df= make_df(folder_path="json_data", score_flag=True)
    df.to_csv('OCR_df.csv', index=False, encoding='utf-8-sig')
    not_tagged_df.to_csv('OCR_test_df.csv', encoding='utf-8-sig', index=False)