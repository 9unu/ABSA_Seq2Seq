import re
def extract_element(sentence):
    # 태깅 문장과 토픽 인식하는 정규식
    pattern = r"(?P<sentence>.+?)\s*\(\s*(?P<topic>.+?)\s*\)"

    # 정규식 컴파일
    regex = re.compile(pattern)
    tag_if = False
    
    # 매칭 및 값 추출
    match = regex.search(sentence)
    if match:
        full_sentence = match.group('sentence').rstrip(':').strip()
        topic = match.group('topic').strip()
        tag_if = True
    else:
        full_sentence="태그 없음"
        topic="토픽 없음"
    return tag_if, full_sentence, topic



from transformers import pipeline
model_name="paust/pko-t5-base"

def generate_text(pipe, text, num_return_sequences=5, max_length=200):
  text = f"{text}"
  out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
  return [x['generated_text'] for x in out]

def run_pipeline(model_path, text):
    nlg_pipeline = pipeline('text2text-generation', model=model_path, tokenizer=model_name, device='cpu')
    return (generate_text(nlg_pipeline, text, num_return_sequences=1, max_length=200)[0])


def bbox_match(total_bbox_list, sentence):
    unmatched_bbox_list = []
    sentence_bbox_list = []
    
    for bbox in total_bbox_list:
        if bbox['text'] in sentence:
            sentence_bbox_list.append(bbox)
        else:
            unmatched_bbox_list.append(bbox)
    
    return unmatched_bbox_list, sentence_bbox_list

