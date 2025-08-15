#!/usr/bin/env python3
"""
sistema_rag.py
Sistema RAG completo para usar com documentos processados no ChromaDB
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json

class SistemaRAG:
    def __init__(self, 
                 chromadb_path: str = "./chromadb_base",
                 collection_name: str = "documentos_word",
                 modelo_llm: str = "llama3",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Inicializa o Sistema RAG
        
        Args:
            chromadb_path: Caminho para a base ChromaDB
            collection_name: Nome da coleção
            modelo_llm: Nome do modelo LLM
            embedding_model: Modelo para embeddings
        """
        
        print("🚀 Inicializando Sistema RAG...")
        
        # 1. Verificar se base ChromaDB existe
        if not os.path.exists(chromadb_path):
            raise Exception(f"❌ Base ChromaDB não encontrada em {chromadb_path}")
        
        # 2. Conectar ao ChromaDB
        print("📚 Conectando à base de documentos...")
        try:
            self.client = chromadb.PersistentClient(path=chromadb_path)
            self.collection = self.client.get_collection(collection_name)
            total_docs = self.collection.count()
            
            if total_docs == 0:
                raise Exception("❌ Base de dados está vazia")
                
            print(f"✓ {total_docs} documentos encontrados na base")
            
        except Exception as e:
            raise Exception(f"❌ Erro ao conectar ChromaDB: {e}")
        
        # 3. Carregar modelo de embeddings
        print("🔢 Carregando modelo de embeddings...")
        try:
            self.embedding_model = SentenceTransformer(f'sentence-transformers/{embedding_model}')
            print("✓ Modelo de embeddings carregado")
        except Exception as e:
            raise Exception(f"❌ Erro ao carregar embeddings: {e}")
        
        # 4. Configurar modelo LLM
        print(f"🤖 Configurando modelo {modelo_llm}...")
        self.llm = self._configurar_modelo_llm(modelo_llm)
        
        if not self.llm:
            raise Exception("❌ Falha ao configurar modelo LLM")
        
        # 5. Configurações RAG
        self.max_context_docs = 5
        self.temperatura = 0.1
        
        print("✅ Sistema RAG pronto!\n")
    
    def _configurar_modelo_llm(self, modelo_nome: str):
        """
        Configura o modelo LLM baseado no tipo
        """
        modelo_nome = modelo_nome.lower()
        
        # Tentar Ollama primeiro (mais comum para uso local)
        if any(name in modelo_nome for name in ["llama", "mistral", "gemma", "phi", "qwen"]):
            return self._configurar_ollama(modelo_nome)
        
        # OpenAI
        elif "gpt" in modelo_nome:
            return self._configurar_openai(modelo_nome)
        
        # Anthropic Claude
        elif "claude" in modelo_nome:
            return self._configurar_anthropic(modelo_nome)
        
        # Tentar como Ollama por padrão
        else:
            return self._configurar_ollama(modelo_nome)
    
    def _configurar_ollama(self, modelo: str):
        """Configura modelo via Ollama (recomendado)"""
        try:
            # Tentar importar bibliotecas necessárias
            try:
                import requests
            except ImportError:
                print("❌ requests não instalado. Execute: pip install requests")
                return None
            
            # Verificar se Ollama está rodando
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    print("❌ Ollama não está respondendo")
                    return None
                    
                modelos_disponiveis = [m['name'] for m in response.json().get('models', [])]
                
                # Verificar se o modelo está instalado
                modelo_encontrado = None
                for m in modelos_disponiveis:
                    if modelo in m or m.startswith(modelo):
                        modelo_encontrado = m
                        break
                
                if not modelo_encontrado:
                    print(f"❌ Modelo {modelo} não encontrado no Ollama")
                    print(f"💡 Modelos disponíveis: {modelos_disponiveis}")
                    print(f"💡 Para instalar: ollama pull {modelo}")
                    return None
                    
            except Exception as e:
                print(f"❌ Erro ao conectar com Ollama: {e}")
                print("💡 Certifique-se que o Ollama está rodando: ollama serve")
                return None
            
            # Configurar cliente Ollama simples
            class OllamaClient:
                def __init__(self, model_name):
                    self.model_name = model_name
                    self.base_url = "http://localhost:11434"
                
                def generate(self, prompt, temperature=0.1):
                    import requests
                    
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": 0.9
                        }
                    }
                    
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json=payload,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            return response.json()['response']
                        else:
                            return f"Erro na API: {response.status_code}"
                            
                    except Exception as e:
                        return f"Erro de conexão: {e}"
            
            # Testar o modelo
            cliente = OllamaClient(modelo_encontrado)
            teste = cliente.generate("Responda apenas 'OK'", temperature=0.0)
            
            if "OK" in teste:
                print(f"✓ Ollama {modelo_encontrado} conectado e funcionando")
                return cliente
            else:
                print(f"⚠️ Ollama conectado mas resposta inesperada: {teste}")
                return cliente  # Retornar mesmo assim
                
        except Exception as e:
            print(f"❌ Erro ao configurar Ollama: {e}")
            return None
    
    def _configurar_openai(self, modelo: str):
        """Configura modelo OpenAI"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OPENAI_API_KEY não encontrada")
                print("💡 Configure: export OPENAI_API_KEY='sua_chave'")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            
            # Testar conexão
            try:
                response = client.chat.completions.create(
                    model=modelo,
                    messages=[{"role": "user", "content": "Responda apenas 'OK'"}],
                    max_tokens=10,
                    temperature=0
                )
                
                if "OK" in response.choices[0].message.content:
                    print(f"✓ OpenAI {modelo} conectado")
                    return client
                    
            except Exception as e:
                print(f"❌ Erro ao testar OpenAI: {e}")
                return None
                
        except ImportError:
            print("❌ openai não instalado. Execute: pip install openai")
            return None
    
    def _configurar_anthropic(self, modelo: str):
        """Configura modelo Anthropic Claude"""
        try:
            import anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("❌ ANTHROPIC_API_KEY não encontrada")
                print("💡 Configure: export ANTHROPIC_API_KEY='sua_chave'")
                return None
            
            client = anthropic.Anthropic(api_key=api_key)
            
            # Testar conexão
            try:
                response = client.messages.create(
                    model=modelo,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Responda apenas 'OK'"}]
                )
                
                if "OK" in response.content[0].text:
                    print(f"✓ Anthropic {modelo} conectado")
                    return client
                    
            except Exception as e:
                print(f"❌ Erro ao testar Anthropic: {e}")
                return None
                
        except ImportError:
            print("❌ anthropic não instalado. Execute: pip install anthropic")
            return None
    
    def buscar_contexto(self, pergunta: str, n_docs: int = None) -> List[Dict[str, Any]]:
        """
        Busca documentos relevantes para usar como contexto
        """
        if n_docs is None:
            n_docs = self.max_context_docs
        
        try:
            # Criar embedding da pergunta
            query_embedding = self.embedding_model.encode(pergunta).tolist()
            
            # Buscar documentos similares
            resultados = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_docs
            )
            
            # Formatar resultados
            contexto = []
            for i in range(len(resultados['documents'][0])):
                contexto.append({
                    'texto': resultados['documents'][0][i],
                    'metadata': resultados['metadatas'][0][i],
                    'id': resultados['ids'][0][i],
                    'similaridade': 1 - resultados['distances'][0][i]
                })
            
            return contexto
            
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []
    
    def criar_prompt_rag(self, pergunta: str, contexto: List[Dict[str, Any]]) -> str:
        """
        Cria prompt otimizado para RAG em português
        """
        # Montar contexto
        contexto_texto = ""
        for i, doc in enumerate(contexto, 1):
            arquivo = doc['metadata'].get('arquivo', 'Documento')
            chunk = doc['metadata'].get('chunk_index', 0)
            similaridade = doc['similaridade']
            
            contexto_texto += f"=== DOCUMENTO {i} ===\n"
            contexto_texto += f"Fonte: {arquivo} (parte {chunk+1}, relevância: {similaridade:.2f})\n"
            contexto_texto += f"Conteúdo: {doc['texto']}\n\n"
        
        # Template do prompt
        prompt = f"""Você é um assistente especializado que responde perguntas baseado exclusivamente nos documentos fornecidos.

DOCUMENTOS DE REFERÊNCIA:
{contexto_texto}

PERGUNTA: {pergunta}

INSTRUÇÕES IMPORTANTES:
1. Use APENAS as informações dos documentos fornecidos acima
2. Se a informação não estiver nos documentos, diga claramente "Não encontrei essa informação nos documentos fornecidos"
3. Sempre cite a fonte usando o formato "Segundo o Documento X"
4. Seja preciso, objetivo e direto
5. Responda em português brasileiro
6. Se múltiplos documentos mencionam o mesmo tema, combine as informações

RESPOSTA:"""

        return prompt
    
    def gerar_resposta(self, prompt: str) -> str:
        """
        Gera resposta usando o modelo LLM
        """
        try:
            # Ollama
            if hasattr(self.llm, 'generate'):
                return self.llm.generate(prompt, temperature=self.temperatura)
            
            # OpenAI
            elif hasattr(self.llm, 'chat'):
                response = self.llm.chat.completions.create(
                    model="gpt-3.5-turbo",  # ou o modelo configurado
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperatura,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            
            # Anthropic
            elif hasattr(self.llm, 'messages'):
                response = self.llm.messages.create(
                    model="claude-3-sonnet-20240229",  # ou o modelo configurado
                    max_tokens=1000,
                    temperature=self.temperatura,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            else:
                return "❌ Tipo de modelo não suportado"
                
        except Exception as e:
            return f"❌ Erro ao gerar resposta: {e}"
    
    def responder_pergunta(self, pergunta: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Responde pergunta usando RAG
        """
        if verbose:
            print(f"🔍 Buscando contexto para: '{pergunta}'")
        
        # 1. Buscar contexto relevante
        contexto = self.buscar_contexto(pergunta)
        
        if not contexto:
            return {
                "pergunta": pergunta,
                "resposta": "❌ Não encontrei documentos relevantes para responder sua pergunta.",
                "contexto_usado": [],
                "fontes": [],
                "erro": "Nenhum contexto encontrado"
            }
        
        if verbose:
            print(f"📚 Encontrados {len(contexto)} documentos relevantes:")
            for i, doc in enumerate(contexto, 1):
                arquivo = doc['metadata'].get('arquivo', 'Desconhecido')
                similaridade = doc['similaridade']
                print(f"  {i}. {arquivo} (similaridade: {similaridade:.2f})")
        
        # 2. Criar prompt
        prompt = self.criar_prompt_rag(pergunta, contexto)
        
        if verbose:
            print("🤖 Gerando resposta...")
        
        # 3. Gerar resposta
        resposta = self.gerar_resposta(prompt)
        
        # 4. Extrair fontes
        fontes = list(set([doc['metadata'].get('arquivo', 'Desconhecido') for doc in contexto]))
        
        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "contexto_usado": contexto,
            "fontes": fontes,
            "erro": None
        }
    
    def modo_chat_interativo(self):
        """
        Modo chat interativo
        """
        print("💬 MODO CHAT RAG")
        print("=" * 50)
        print("Digite suas perguntas sobre os documentos")
        print("Comandos especiais:")
        print("  'info' - informações sobre a base")
        print("  'config' - ver configurações")
        print("  'clear' - limpar tela")
        print("  'sair' - encerrar")
        print("-" * 50)
        
        while True:
            try:
                pergunta = input("\n🙋 Você: ").strip()
                
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Até logo!")
                    break
                
                elif pergunta.lower() == 'info':
                    total = self.collection.count()
                    print(f"📊 Base de dados: {total} chunks")
                    
                    # Mostrar alguns arquivos
                    amostra = self.collection.get(limit=5)
                    arquivos = set()
                    for metadata in amostra['metadatas']:
                        arquivos.add(metadata.get('arquivo', 'Desconhecido'))
                    
                    print(f"📚 Arquivos: {', '.join(list(arquivos)[:3])}")
                    continue
                
                elif pergunta.lower() == 'config':
                    print(f"⚙️ Configurações:")
                    print(f"  - Documentos de contexto: {self.max_context_docs}")
                    print(f"  - Temperatura: {self.temperatura}")
                    continue
                
                elif pergunta.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                elif not pergunta:
                    continue
                
                # Processar pergunta
                print("\n🤖 Assistente: ", end="")
                resultado = self.responder_pergunta(pergunta, verbose=False)
                
                if resultado['erro']:
                    print(f"❌ {resultado['erro']}")
                else:
                    print(resultado['resposta'])
                    
                    # Mostrar fontes
                    if resultado['fontes']:
                        print(f"\n📚 Fontes: {', '.join(resultado['fontes'])}")
                        
            except KeyboardInterrupt:
                print("\n\n👋 Interrompido pelo usuário. Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
    
    def avaliar_sistema(self, perguntas_teste: List[str] = None) -> Dict[str, Any]:
        """
        Avalia o sistema RAG
        """
        if perguntas_teste is None:
            perguntas_teste = [
                "Qual é o assunto principal dos documentos?",
                "Quais são as principais conclusões?",
                "Que metodologia foi utilizada?",
                "Existem recomendações específicas?"
            ]
        
        print("🧪 AVALIAÇÃO DO SISTEMA RAG")
        print("=" * 50)
        
        resultados = []
        
        for i, pergunta in enumerate(perguntas_teste, 1):
            print(f"\n[{i}/{len(perguntas_teste)}] {pergunta}")
            
            resultado = self.responder_pergunta(pergunta, verbose=False)
            
            # Calcular score simples
            if resultado['erro']:
                score = 0.0
                print(f"❌ Erro: {resultado['erro']}")
            else:
                score = min(len(resultado['contexto_usado']) / self.max_context_docs, 1.0)
                print(f"✓ Score: {score:.2f}")
            
            resultados.append({
                'pergunta': pergunta,
                'score': score,
                'tem_contexto': len(resultado.get('contexto_usado', [])) > 0,
                'tem_resposta': resultado['resposta'] is not None
            })
        
        # Calcular métricas finais
        score_medio = sum(r['score'] for r in resultados) / len(resultados)
        taxa_contexto = sum(1 for r in resultados if r['tem_contexto']) / len(resultados)
        taxa_resposta = sum(1 for r in resultados if r['tem_resposta']) / len(resultados)
        
        print(f"\n📊 MÉTRICAS FINAIS:")
        print(f"Score médio: {score_medio:.2f}")
        print(f"Taxa de encontrar contexto: {taxa_contexto:.1%}")
        print(f"Taxa de gerar resposta: {taxa_resposta:.1%}")
        
        if score_medio >= 0.7:
            print("✅ Sistema funcionando bem!")
        elif score_medio >= 0.4:
            print("⚠️ Sistema funcionando, mas pode melhorar")
        else:
            print("❌ Sistema precisa de ajustes")
        
        return {
            'score_medio': score_medio,
            'taxa_contexto': taxa_contexto,
            'taxa_resposta': taxa_resposta,
            'resultados': resultados
        }


def exemplo_uso_ollama():
    """
    Exemplo usando Ollama com Llama 3
    """
    print("🦙 EXEMPLO: Sistema RAG com Llama 3")
    print("=" * 50)
    
    try:
        # Inicializar sistema
        rag = SistemaRAG(
            modelo_llm="llama3",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Fazer algumas perguntas de teste
        perguntas = [
            "Qual é o tema principal dos documentos?",
            "Quais são os pontos mais importantes?",
            "Existe alguma conclusão específica?"
        ]
        
        for pergunta in perguntas:
            print(f"\n❓ {pergunta}")
            resultado = rag.responder_pergunta(pergunta, verbose=False)
            
            if resultado['erro']:
                print(f"❌ {resultado['erro']}")
            else:
                print(f"🤖 {resultado['resposta']}")
                if resultado['fontes']:
                    print(f"📚 Fontes: {', '.join(resultado['fontes'])}")
        
        # Avaliar sistema
        print(f"\n" + "=" * 50)
        rag.avaliar_sistema()
        
        # Modo interativo
        print(f"\n💬 Iniciando modo chat...")
        rag.modo_chat_interativo()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n💡 Soluções:")
        print("1. Certifique-se que processou os documentos primeiro")
        print("2. Verifique se Ollama está rodando: ollama serve")
        print("3. Instale Llama 3: ollama pull llama3")


def menu_principal():
    """
    Menu principal do sistema RAG
    """
    while True:
        print("\n" + "=" * 50)
        print("🤖 SISTEMA RAG - CONSULTA DE DOCUMENTOS")
        print("=" * 50)
        print("1. 🚀 Iniciar sistema RAG (Ollama)")
        print("2. 💰 Usar OpenAI (GPT)")
        print("3. 🧪 Avaliar sistema")
        print("4. ❓ Ajuda e configuração")
        print("5. 🚪 Sair")
        print("-" * 50)
        
        escolha = input("Escolha uma opção (1-5): ").strip()
        
        if escolha == "1":
            try:
                rag = SistemaRAG(modelo_llm="llama3")
                rag.modo_chat_interativo()
            except Exception as e:
                print(f"❌ Erro: {e}")
                
        elif escolha == "2":
            api_key = input("Digite sua chave OpenAI (ou Enter para usar variável de ambiente): ").strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            
            try:
                rag = SistemaRAG(modelo_llm="gpt-3.5-turbo")
                rag.modo_chat_interativo()
            except Exception as e:
                print(f"❌ Erro: {e}")
                
        elif escolha == "3":
            try:
                modelo = input("Qual modelo usar? (llama3/gpt-3.5-turbo): ").strip() or "llama3"
                rag = SistemaRAG(modelo_llm=modelo)
                rag.avaliar_sistema()
            except Exception as e:
                print(f"❌ Erro: {e}")
                
        elif escolha == "4":
            print("\n📖 AJUDA E CONFIGURAÇÃO")
            print("-" * 30)
            print("PRÉ-REQUISITOS:")
            print("1. Documentos processados (execute processar_documentos.py primeiro)")
            print("2. Modelo LLM configurado:")
            print("\n   OLLAMA (Recomendado - Gratuito):")
            print("   - Instalar: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   - Modelo: ollama pull llama3")
            print("   - Iniciar: ollama serve")
            print("\n   OPENAI (Pago):")
            print("   - Chave API: export OPENAI_API_KEY='sua_chave'")
            print("   - Instalar: pip install openai")
            print("\nDEPENDÊNCIAS:")
            print("pip install chromadb sentence-transformers requests")
            
        elif escolha == "5":
            print("\n👋 Até logo!")
            break
            
        else:
            print("\n❌ Opção inválida.")


if __name__ == "__main__":
    # Verificar se base ChromaDB existe
    if not os.path.exists("./chromadb_base"):
        print("❌ Base ChromaDB não encontrada!")
        print("💡 Execute primeiro: python processar_documentos.py")
        exit(1)
    
    # Verificar dependências
    try:
        import chromadb
        import sentence_transformers
        import requests
    except ImportError as e:
        print("❌ Dependência não encontrada!")
        print("Execute: pip install chromadb sentence-transformers requests")
        exit(1)
    
    # Executar
    print("🎯 Bem-vindo ao Sistema RAG!")
    menu_principal()