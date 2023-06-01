from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
                         AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer
import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
                             RandomSampler, SequentialSampler

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

MODEL           = 'gpt2' #{gpt2, gpt2-medium, gpt2-large, gpt2-xl}

MAXLEN          = 1024  #{768, 1024, 1280, 1600}

def get_tokenizer(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    #----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)
    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.cuda()
    return model

tokenizer = get_tokenizer(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path='C:/Users/alosh/OneDrive/Desktop/VSCODE/amazon_reviews/pytorch_model.bin')

print("Enter something: ")
title = input() 

prompt = SPECIAL_TOKENS['bos_token'] + title + SPECIAL_TOKENS['sep_token'] 
         
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
device = torch.device("cuda")
generated = generated.to(device)

model.eval();

# Top-p (nucleus) text generation (10 samples):
# Beam-search text generation:
# sample_outputs = model.generate(generated, 
#                                 do_sample=True,   
#                                 max_length=MAXLEN,                                                      
#                                 num_beams=5,
#                                 repetition_penalty=5.0,
#                                 early_stopping=True,      
#                                 num_return_sequences=1
#                                 )

# for i, sample_output in enumerate(sample_outputs):
#     text = tokenizer.decode(sample_output, skip_special_tokens=True)
#     a = len(title) 
#     print("{}: {}\n\n".format(i+1,  text[a:]))

#Generating raw text with GPT-2
tokenizer = get_tokenizer()
model = get_model(tokenizer)

prompt = title

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
device = torch.device("cuda")
generated = generated.to(device)

model.eval()
sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                max_length=MAXLEN,                                                      
                                num_beams=5,
                                repetition_penalty=5.0,
                                early_stopping=False,      
                                num_return_sequences=1
                                )

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))