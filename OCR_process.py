from OCR_process_func import *
import os
import json
import copy

folder_path='OCR_json'
file_list = sorted(os.listdir(folder_path))
save_folder_path="OCR_processed_json"

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

for file_name in file_list:
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8-sig') as file:
            df=[json.load(file)]

        matched_bbox_list = []
        for index, row in enumerate(df):
            for i in range(len(row)):
                splited_list = row[i][0].split("\n")
                if type(row[i][1]) == list:
                    total_bbox_list = row[i][2:]
                else:
                    total_bbox_list = row[i][1:]
                total_bbox_list_to_save = copy.deepcopy(total_bbox_list)
                # 한 이미지의 문장마다 태그 정보 필요함    
                tag_result_list=[]
                for sentence in splited_list:
                    # 분할된 문장의 단어와 이미지 내의 단어와 매칭 --> bbox 묶음 (sentence_bbox_list)
                    unmatched_bbox_list, sentence_bbox_list = bbox_match(total_bbox_list, sentence)
                    # 해당 문장의 태그 정보
                    model_tag=run_pipeline('OCR_model.pth', sentence)
                    tag_info, tagged_sentence, topic = extract_element(model_tag)
                    tag_result={}
                    # 태그가 있으면 토픽과 함께 dict로 저장 
                    if tag_info:
                        tag_result['text']=tagged_sentence
                        tag_result['topic']=topic
                        tag_result['start_pos']=0
                        tag_result['end_pos']=0
                        
                        tagged_bbox_list=[]
                        for tag in sentence_bbox_list:
                            if tag['text'] in tagged_sentence:
                                tagged_bbox_list.append(tag['bbox'])
                        tag_result['bbox']=tagged_bbox_list
                    else:
                        # 태그가 없는 경우는 json에 저장안함
                        continue
                    tag_result_list.append(tag_result)
                    # 다음 문장에 대해, 앞에서 매칭되지 않고 남은 bbox 리스트에서 다시 매칭하도록함
                    total_bbox_list = unmatched_bbox_list
                save_list=[row[i][0], tag_result_list]
                for bbox in total_bbox_list_to_save:
                    save_list.append(bbox)
                matched_bbox_list.append(save_list)
        
        with open(os.path.join(save_folder_path, file_name), 'w', encoding='utf-8-sig') as f:
            json.dump(matched_bbox_list, f, ensure_ascii=False)