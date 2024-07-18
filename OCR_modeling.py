from sklearn.model_selection import train_test_split
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback)
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import pandas as pd
import shutil

def modeling(csv_path):
    print(torch.cuda.is_available())
    # origin 이랑 다른 말투를 매칭
    df=pd.read_csv(csv_path)
    # print(len(df))
    # df_not=df[df['tag_info']=="태그 없음"]
    # print(df_not)
    # df=df[df['tag_info']!="태그 없음"]
    '''태그없음을 따로 저장'''
    # print(len(df))
    df, df_test = train_test_split(df, test_size=0.2, random_state=42)
    # df_test=pd.concat([df_not, df_test], axis=0)
    # df_not.to_csv("not_tagged_test.csv", encoding='utf-8-sig', index=False)
    df_test.to_csv("OCR_test_df.csv", encoding='utf-8-sig', index=False)
    model_name="paust/pko-t5-base"
    # 모델 로드
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    
# 학습 데이터 만드는 class
    class TextStyleTransferDataset(Dataset):
        def __init__(self,
                    df: pd.DataFrame,
                    tokenizer: Tokenizer
                    ):
            self.df = df
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            row = self.df.iloc[index, :]
            sentence = row['sentence']  # 'content' 열 값
            sentence_aspect = row['tag_info']  # 'text_aspect' 열 값
            encoder_text = f"{sentence}"
            decoder_text = f"{sentence_aspect}{self.tokenizer.eos_token}"
            model_inputs = self.tokenizer(encoder_text, max_length=64, truncation=True)

            with self.tokenizer.as_target_tokenizer():
                labels = tokenizer(decoder_text, max_length=64, truncation=True)
                model_inputs['labels'] = labels['input_ids']
            # del model_inputs['token_type_ids']

            return model_inputs
    
    
    # 데이터 분할

    # 문체 변환용으로 데이터 변환
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(len(df_train), len(df_test))

    train_dataset = TextStyleTransferDataset(
        df_train,
        tokenizer
    )
    test_dataset = TextStyleTransferDataset(
        df_test,
        tokenizer
    )

    model =T5ForConditionalGeneration.from_pretrained(model_name)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model
    )

    directory_to_delete = "saved_model"

    training_args = Seq2SeqTrainingArguments(
                    directory_to_delete,
                    evaluation_strategy = "epoch",
                    save_strategy = "epoch",
                    eval_steps = 10,
                    load_best_model_at_end = True,
                    per_device_train_batch_size=10,
                    per_device_eval_batch_size=10,
                    gradient_accumulation_steps=2,
                    weight_decay=0.01,
                    save_total_limit=1,
                    num_train_epochs=20,
                    predict_with_generate=True,
                    fp16=False,
            )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 모델 학습
    trainer.train()

    # trainer.save_model("without_score_model.pth")
    if os.path.exists(directory_to_delete):
        shutil.rmtree(directory_to_delete)
    return trainer
        
"""함수 호출"""
if __name__ == "__main__":
    # without_score_model=modeling(r'C:\Users\KHU\Desktop\tagging_seq2seq\sentenced_df_without_score.csv')
    # without_score_model.save_model("without_score_model.pth")
    with_score_model=modeling(r'C:\Users\KHU\Desktop\tagging_seq2seq\sentenced_df_OCR.csv')
    with_score_model.save_model("OCR_model.pth")