### BERT-TextAnalysis
1. Sentiment analysis
2. Fill masking
3. fine-tuning

#### Sentiment analysis
사전 학습 모델: `nlptown/bert-base-multilingual-uncased-sentiment` 모델 사용

```py
def sentiment_pipeline():
    """
        감정 분석 파이프라인 데모
        - NLPTown/bert-base-multilingual-uncased-sentiment 모델 사용
    """
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
        truncation=True  # 최대 길이 초과 시 자르기
    )
    dataset = load_dataset("imdb")
    examples = dataset["train"]["text"][:5]

    print("Sentiment Analysis Pipeline")
    print("----------------------------------------------------------")
    print(examples)
    for text in examples:
        result = sentiment_analyzer(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")
```

#### Fill masking
사전 학습 모델: bert-base-multilingual-cased 모델 사용
```py
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
```
