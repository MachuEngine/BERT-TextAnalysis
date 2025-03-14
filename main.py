import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForMaskedLM,
    pipeline,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

"""
    *. COMMANDS
    # pip install torch transformers datasets
    # python main.py
    ----------------------------------------------------------
    *. Library Descriptions
    - BERT 모델을 활용한 텍스트 분석 파이프라인 데모
        1. transformers 라이브러리의 주요 클래스/함수:
            BertTokenizer: 텍스트를 BERT 입력 형식에 맞게 토큰화/인코딩하는 클래스.
            BertForSequenceClassification: 문장 분류(감정 분석, 스팸 분류 등)용 BERT 모델 클래스.
            BertForMaskedLM: 마스킹된 토큰을 예측하는 언어 모델(BERT) 클래스.
            pipeline: 특정 태스크(감정 분석, 마스킹 예측 등)를 간편하게 수행하게 해주는 높은 수준의 인터페이스.
            Trainer: 파인튜닝(fine-tuning) 과정을 쉽게 구현할 수 있도록 도와주는 클래스.
            TrainingArguments: Trainer가 사용할 하이퍼파라미터나 설정을 담는 클래스.

        2. datasets 라이브러리의 주요 함수:
            load_dataset: 다양한 데이터셋을 쉽게 불러올 수 있도록 도와주는 함수. (huggingface.co/datasets 참고)
            Hugging Face의 Datasets 라이브러리이며, load_dataset 함수로 유명 데이터셋(IMDB, SQuAD, GLUE 등)을 불러올 수 있음.
"""

def sentiment_pipeline():
    """
        감정 분석 파이프라인 데모
        - NLPTown/bert-base-multilingual-uncased-sentiment 모델 사용
    """
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    examples = [
        "This movie was really fun!",
        "It was worse than I expected. Waste of money.",
        "Wow... It was amazing. I'd like to watch it twice."
    ]
    for text in examples:
        result = sentiment_analyzer(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")

def bert_fine_tuning():
    """
    BERT 모델 파인튜닝 데모
    bert-base-multilingual-cased 모델 사용
    ----------------------------------------------------------
    - IMDB 데이터를 로드해 토크나이징 → PyTorch 형식으로 변환
    - BERT 분류 모델(이진 감정 분석)을 준비
    - Trainer로 학습(파인튜닝) 수행
    - 학습된 모델 저장 및 평가 결과 출력
    """
    # BERT 멀티 언어 모델 이름 지정
    model_name = "bert-base-multilingual-cased"
    
    # BERT 모델과 토크나이저 객체 생성 및 분류 모델 준비
    # num_labels=2: 이진 분류(긍정/부정)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # IMDB 데이터셋 로드 (Hugging Face Datasets 라이브러리 활용)
    dataset = load_dataset("imdb")
    
    # ----------------------------------------------------------
    # examples는 dataset.map(tokenize_fn, batched=True) 를 호출할 때,
    #       map 메서드가 내부적으로 나누어 전달해주는 데이터 배치(batch).
    # dataset.map(...)를 호출하면, 각 샘플(레코드)의 컬럼들을 모아 
    #       examples라는 딕셔너리 형태로 tokenize_fn에 전달함.
    # batched=True 옵션 때문에, 한 번에 여러 샘플이 묶여서 전달되어, 
    #       examples["text"]는 리스트 형태의 텍스트들.
    # ----------------------------------------------------------
    # truncation=True: 문장이 길 경우 자르기
    # padding="max_length": 문장이 짧을 경우 패딩 추가
    # max_length=128: 최대 길이 128로 설정
    # ----------------------------------------------------------
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    # dataset.map(...)을 통해 examples의 텍스트를 토큰화한다. 
    encoded_dataset = dataset.map(tokenize_fn, batched=True)

    # 토큰화된 데이터셋의 컬럼 이름을 "label"에서 "labels"로 변경
    # 이유: Trainer 클래스에서 정답 레이블 컬럼을 찾을 때 "labels"로 찾기 때문
    encoded_dataset = encoded_dataset.rename_column("label", "labels")

    # 데이터셋의 포맷을 PyTorch로 변경하고 컬럼 이름을 input_ids, attention_mask, labels로 설정
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 훈련 데이터셋과 테스트 데이터셋으로 나누기
    train_dataset = encoded_dataset["train"]
    test_dataset = encoded_dataset["test"]

    # Trainer arguments 설정
    # evaluation_strategy="epoch": 에폭이 끝날 때마다 평가 수행
    # per_device_train_batch_size=8: 한 번에 학습할 배치 크기
    # num_train_epochs=1: 총 학습 에폭 수
    # save_steps=5000: 5000번 스텝마다 모델 저장
    # save_total_limit=2: 2개 모델까지만 저장
    training_args = TrainingArguments(
        output_dir="./bert-sentiment-model",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        save_steps=5000,
        save_total_limit=2
    )

    # 학습 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # 학습 수행
    trainer.train()

    # 학습된 모델 저장
    trainer.save_model("./bert-sentiment-model")

    # 테스트 데이터셋으로 평가 수행
    eval_result = trainer.evaluate()
    print(f"Evaluation result on test set: {eval_result}")

def fill_mask():
    """
        마스킹된 토큰을 예측하는 언어 모델(BERT) 데모
        - bert-base-multilingual-cased 모델 사용
    """
    mask_filler = pipeline("fill-mask", model="bert-base-multilingual-cased")
    masked_text = "This movie is absolutely [MASK]!"
    predictions = mask_filler(masked_text)
    print(f"Input: {masked_text}")
    print("Predictions:")
    for pred in predictions:
        print(
            f"- {pred['sequence']} "
            f"(score={pred['score']:.4f}, token={pred['token_str']})"
        )

if __name__ == "__main__":
    sentiment_pipeline()
    # bert_fine_tuning() // 튜닝 시간 제약 문제로, 사용하려면 주석 해제 후 실행
    fill_mask()
