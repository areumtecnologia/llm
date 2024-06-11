# Developed by Renan Teixeira Moreira (dev@areum.com.br)
# Licensed by Áreum Tecnologia

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig, Trainer, TrainingArguments
import torch
import os
import gc
import re
class LLMQ4:
    # Load 6.188 GB
    def __init__(self, repo_id, quantized=True, trust_remote_code=False, attention_implementation=None):
        # Definindo a configuração para evitar fragmentação
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Model name
        model_name = repo_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_quantized = quantized
        self.tokenizer, self.model = self.initialize_model(model_name, quantized, trust_remote_code, attention_implementation)

    def initialize_model(self, model_name, quantized, trust_remote_code, attention_implementation):
        model = None
        if quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, torch_dtype=torch.bfloat16, quantization_config=bnb_config, trust_remote_code=trust_remote_code)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            # Configure attention implementation after loading the model
            if attention_implementation and hasattr(model.config, 'attn_implementation'):
                model.config.attn_implementation = attention_implementation
        # Tokenizer initialization
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print('GPU alocada ', torch.cuda.memory_allocated()/ (1024**2))
        return tokenizer, model
    
    def prompt(self, context):
        
        # CONTEXT DEVE TER CONFIGURACAO POR ROLE
        text = self.tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=38000,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=[
                self.tokenizer.eos_token_id,
            ],
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Limpeza de memória

        torch.cuda.empty_cache()
        gc.collect()
        return response
    
    def prepare_dataset(self, prompts, responses):
        # Tokenizar os prompts e as respostas
        train_encodings = self.tokenizer(prompts, truncation=True, padding=True)
        train_labels = self.tokenizer(responses, truncation=True, padding=True)

        # Garantir que os tamanhos de lote coincidam
        input_ids = train_encodings['input_ids']
        labels = train_labels['input_ids']

        max_length = max(len(ids) for ids in input_ids)
        for i in range(len(labels)):
            if len(labels[i]) < max_length:
                labels[i] = labels[i] + [-100] * (max_length - len(labels[i]))
            else:
                labels[i] = labels[i][:max_length]

        class FineTuneDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels["input_ids"][idx])
                return item

            def __len__(self):
                return len(self.labels["input_ids"])

        return FineTuneDataset(train_encodings, train_labels)

    def train(self, train_dataset, output_dir="./results", num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=4):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            fp16=True  # Habilitar precisão mista
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
    
    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path, quantized=True, trust_remote_code=False, attention_implementation=None):
        self.tokenizer, self.model = self.initialize_model(path, quantized, trust_remote_code, attention_implementation)
   
# Exemplo de uso
# llama_chatqa = LLMQ4('Qwen/Qwen2-0.5B-Instruct')
# history = [
#     {"role": "system", "content": "Your area Hum, a helpful assistant created by Áreum Tecnologia."},
#     # Skills SystemPrompts
#     {"role": "system", "content": "You can access the internet and process the information collected."},
#     {"role": "system", "content": "You can create and analyse images."},
#     {"role": "system", "content": "You can listen and process audio."},
#     {"role": "system", "content": "You can read and process PDF files."},
#     {"role": "user", "content": "Quem é você?"},
# ]
# response = llama_chatqa.prompt(history)
# print(response)