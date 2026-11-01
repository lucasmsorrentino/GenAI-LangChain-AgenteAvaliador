# Agente Avaliador

Exercício da Aula 02 de GenAI, Prof. Vinícius Fülber Garcia, a partir do [esqueleto do professor](https://github.com/fulbergarciaufpr/GenAI-LangChain-AgenteAvaliador).

O script busca as reviews da API Codante e classifica cada texto como positivo ou negativo usando `ChatPromptTemplate`, `ChatOllama` e `StrOutputParser`, na chain `prompt | llm | output_parser`.

O extra está incluído: conforme o sentimento, outra chain gera uma sugestão de resposta. Tudo é exibido no terminal.

## Executar

Com Python 3.12 e Ollama instalados, abra o PowerShell na pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3.5:4b
.\.venv\Scripts\python.exe agente_avaliacoes.py
```

O Ollama precisa estar em execução. Se necessário, use `ollama serve` em outro terminal.

O modelo `qwen3.5:4b` já estava instalado no computador usado para desenvolver o exercício. Ele é definido diretamente no script. Não é necessária chave de API.

A classificação é feita pelo texto da review. As respostas são sugestões geradas pela LLM e não são publicadas na API.
