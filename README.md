# Agente Avaliador

Exercício da Aula 02 de GenAI, Prof. Vinícius Fülber Garcia, a partir do [esqueleto do professor](https://github.com/fulbergarciaufpr/GenAI-LangChain-AgenteAvaliador).

O script busca as reviews da API Codante e classifica cada texto como positivo ou negativo usando `ChatPromptTemplate`, `ChatOllama` e `StrOutputParser`, na chain `prompt | llm | output_parser`. Também exibe a confiança estimada pela LLM.

O extra está incluído: conforme o sentimento, outra chain gera uma sugestão de resposta. Tudo é exibido no terminal.

## Executar

Com Python 3.12 e Ollama instalados, abra o PowerShell na pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3.5:0.8b
.\.venv\Scripts\python.exe agente_avaliacoes.py
```

O Ollama precisa estar em execução. Se necessário, use `ollama serve` em outro terminal.

O modelo `qwen3.5:0.8b` é definido diretamente no script. Não é necessária chave de API. A confiança é uma estimativa do modelo e não uma medida de acurácia validada.

A classificação é feita pelo texto da review. As respostas são sugestões geradas pela LLM e não são publicadas na API. Classificações e sugestões podem conter erros e devem ser revisadas.

Verificação realizada: execução real com as dez reviews da API, retornando sete positivas, três negativas, confiança validada pelo Pydantic e dez sugestões de resposta, sem falhas de execução. O modelo leve ainda pode incluir detalhes não mencionados pelo cliente nas sugestões; revise o texto antes de usá-lo.
