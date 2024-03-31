import os
import sentencepiece
import protobuf

import huggingface_hub
from huggingface_hub import hf_hub_download

HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

model_id = "lmsys/fastchat-t5-3b-v1.0"



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)


pipeline("Help me schedule my Day")