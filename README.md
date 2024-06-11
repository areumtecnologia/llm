# llm
Class to handler T2T LLMs with train

# USAGE:

from classes.TextToTextModels import LLMQ4

## Inicializar o modelo
llmq4 = LLMQ4('Qwen/Qwen2-0.5B', quantized=False)

## Preparar dados de treinamento
prompts = ["What is your name?", "Who created you?"]
responses = ["My name is Sentinela.", "I was created by the development team of the company Áreum Tecnologia."]
train_dataset = llmq4.prepare_dataset(prompts, responses)

## Treinar o modelo com tamanho de lote reduzido e precisão mista
llmq4.train(train_dataset, num_train_epochs=500)  # Reduzindo o tamanho do lote

## Salvar o modelo treinado
llmq4.save_model("./trained_model")

## Carregar o modelo treinado
llmq4.load_model("./trained_model")

## Fazer uma inferência com o modelo treinado

response = llmq4.prompt([{"role": "user", "content": "Quem criou voce?"}])
print(response)
