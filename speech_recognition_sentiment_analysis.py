import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

def transcribe_audio(audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load Whisper ASR model
    whisper_model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)

    processor = AutoProcessor.from_pretrained(whisper_model_id)

    pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model_id,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    )

    result = pipe(audio_path)
    return result["text"]

def analyze_sentiment(text):
    # Load sentiment analysis model
    sentiment_model = "nlptown/bert-base-multilingual-uncased-sentiment"
    sentiment_analysis = pipeline(task="sentiment-analysis", model=sentiment_model)

    # Analyze sentiment
    sentiment_result = sentiment_analysis(text)[0]
    return sentiment_result

# Optional: summarizing text using model from Happy Face
def summarize_text(text):
    # Load summarization model
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    # Summarize text
    summary = summarizer(text, max_length=40, min_length=30, do_sample=False)

    return summary[0]['summary_text']

def main():

    # Get audio input from the user
    audio_path = input("Enter the path to the audio file: ")

    # Transcribe audio
    transcribed_text = transcribe_audio(audio_path)
    print("Transcribed Text:", transcribed_text)

    # Analyze sentiment
    sentiment_result = analyze_sentiment(transcribed_text)
    print(f"Sentiment Analysis: {sentiment_result}")

    # Summarize text
    summarized_result = summarize_text(transcribed_text)
    print("Summarized Text:", summarized_result)

if __name__ == "__main__":
    main()