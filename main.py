# Use o código do segundo script (sistema RAG)
from src.sistema_rag import ModeloRAG

# Inicializar com seu modelo escolhido
rag = ModeloRAG(modelo_llm="llama3")  # ou "gpt-3.5-turbo"

# Fazer uma pergunta
resultado = rag.responder_pergunta("Qual é o tema principal dos documentos?")
print(resultado['resposta'])