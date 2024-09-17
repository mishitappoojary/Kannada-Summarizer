from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBARTSS", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBARTSS")

def summarize_text(text, language_code="<2kn>", max_length=100, min_length=30):
    # Prepare the input text in the format required by the model
    input_text = f"{text} </s> {language_code}"
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=tokenizer.convert_tokens_to_ids(language_code)
    )
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return summary
