import json
import os
import pandas as pd

folder_path = "json_data"

file_list = os.listdir(folder_path)

total_df=pd.DataFrame()
for file_name in file_list:
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
        df = pd.json_normalize(data)
        '''태깅이 아예안된 파일은 스킵'''
        if 'our_topics' not in df.columns:
            continue
        '''태깅 안된 빈 리스트 제거'''
        for index, row in df.iterrows():
            if row['our_topics']==[]:
                df.drop(index=index, inplace=True)
        
        '''태깅된 부분과 aspect를 1:1 매칭'''
        df['text_aspect']=""
        for index, row in df.iterrows():
            total_str=""
            for tag in row['our_topics']:
                text=tag['text']
                if text=='':
                    continue
                topic=tag['topic']
                tag['positive_yn'] = tag['positive_yn'].replace('Y', '긍정').replace('N', '부정')
                sentiment=tag['positive_yn']
                sentiment_score=tag['sentiment_scale']
                try:
                    topic_score=tag['topic_score']
                except:
                    topic_score='1'
                total_str+=f"{text} : ({topic}[{topic_score}])({sentiment}[{sentiment_score}]) // " 
            if total_str.rstrip() != "":
                df.loc[index, 'text_aspect'] = total_str.rstrip(' // ')
            selected_df = df[['content', 'text_aspect']]
            selected_df=selected_df[selected_df['text_aspect']!=""]
        # total_df에 데이터프레임 추가
        total_df = pd.concat([total_df, selected_df], ignore_index=True)
if __name__ == "__main__":
    total_df.to_csv('seq2seq_data.csv', index=False, encoding='utf-8-sig')