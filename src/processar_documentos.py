#!/usr/bin/env python3
"""
processar_documentos.py
Script para processar documentos DOCX e adicionar ao ChromaDB
"""

import os
import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DADOS = "./chromadb_base"  
PASTA_DOCUMENTOS = "./documentos"
COLECAO = "documentos_word"             

def extrair_texto_docx(caminho_arquivo):
    """
    Extrai texto de arquivo DOCX
    """
    try:
        from docx import Document
        
        doc = Document(caminho_arquivo)
        
        texto_paragrafos = extrair_paragrafos(doc)
        texto_tabelas = extrair_tabelas(doc)
        
        return (texto_paragrafos + texto_tabelas).strip()
        
    except ImportError:
        print("Erro: python-docx não instalado. Execute: pip install python-docx")
        return None
        
    except Exception as e:
        print(f"Erro ao extrair texto de {caminho_arquivo}: {e}")
        return None


def extrair_paragrafos(doc):
    """
    Extrai todos os parágrafos do documento
    """
    paragrafos = []
    
    for paragrafo in doc.paragraphs:
        if paragrafo.text.strip():
            paragrafos.append(paragrafo.text)
    
    return "\n".join(paragrafos) + "\n" if paragrafos else ""


def extrair_tabelas(doc):
    """
    Extrai todas as tabelas do documento
    """
    if not doc.tables:
        return ""
    
    linhas_tabela = []
    
    for tabela in doc.tables:
        for linha in tabela.rows:
            linha_texto = " | ".join(celula.text for celula in linha.cells)
            
            if linha_texto.strip():
                linhas_tabela.append(linha_texto)
    
    return "\n".join(linhas_tabela) + "\n" if linhas_tabela else ""


def dividir_texto_em_chunks(texto, tamanho=1000, sobreposicao=200):
    """
    Divide texto em chunks menores com sobreposição
    """
    if not texto or len(texto) < 50:
        return []
    
    chunks = []
    inicio = 0
    
    while inicio < len(texto):
        fim = inicio + tamanho
        
       
        if fim < len(texto):
            quebra_paragrafo = texto.rfind('\n\n', inicio, fim)
            if quebra_paragrafo > inicio + tamanho * 0.5:
                fim = quebra_paragrafo + 2
            else:
                # Procurar por fim de frase
                quebra_frase = texto.rfind('. ', inicio, fim)
                if quebra_frase > inicio + tamanho * 0.7:
                    fim = quebra_frase + 2
        
        chunk = texto[inicio:fim].strip()
        
        if len(chunk) > 50:
            chunks.append(chunk)
            
        inicio = max(0, fim - sobreposicao)
    
    return chunks


def verificar_pasta_de_documentos():
    """
    Verifica pasta e retorna lista de arquivos .docx válidos
    """
    if not os.path.exists(PASTA_DOCUMENTOS):
        print(f"Erro: Pasta '{PASTA_DOCUMENTOS}' não encontrada")
        print(f"Crie a pasta com: mkdir {PASTA_DOCUMENTOS}")
        return []
    
    pasta = Path(PASTA_DOCUMENTOS)
    arquivos_docx = []
    
    for padrao in ["*.docx", "*.DOCX"]:
        arquivos_docx.extend(pasta.glob(padrao))
    
    arquivos_docx = [f for f in arquivos_docx if not f.name.startswith("~$")]
    
    if not arquivos_docx:
        print(f"Erro: Nenhum arquivo .docx encontrado em '{PASTA_DOCUMENTOS}'")
        return []
    
    return arquivos_docx


def criar_chromadb(base_dados):
    """
    Cria cliente ChromaDB persistente
    """
    try:
        client = chromadb.PersistentClient(path=base_dados)
        return client
        
    except PermissionError as e:
        print(f"Erro: Permissão negada ao acessar banco de dados: {e}")
        return None
        
    except Exception as e:
        print(f"Erro ao configurar ChromaDB: {e}")
        return None


def criar_colecao(client, colecao):
    """
    Obtém coleção existente ou cria nova
    """
    if not client or not colecao:
        return None
        
    try:
        collection = client.get_collection(colecao)
        return collection
        
    except NotFoundError:
        try:
            collection = client.create_collection(colecao)
            return collection
        except Exception as e:
            print(f"Erro ao criar coleção: {e}")
            return None
            
    except Exception as e:
        print(f"Erro ao acessar coleção: {e}")
        return None


def carregar_modelo_embedding():
    """
    Carrega modelo de embeddings
    """
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return embedding_model
        
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        print("Execute: pip install sentence-transformers")
        return None


def criar_metadados(chunks, arquivo_path, texto):
    """
    Cria metadados para cada chunk
    """
    metadados = []
    
    for i, chunk in enumerate(chunks):
        metadados.append({
            "arquivo": arquivo_path.name,
            "caminho_completo": str(arquivo_path.absolute()),
            "chunk_index": i,
            "total_chunks": len(chunks),
            "tamanho_chunk": len(chunk),
            "palavras_chunk": len(chunk.split()),
            "tamanho_arquivo_original": len(texto)
        })
    
    return metadados


def chunks_already_exists(collection, ids):
    """
    Verifica se chunks já existem na coleção
    """
    try:
        existentes = collection.get(ids=ids)
        return bool(existentes['ids'])
        
    except Exception:
        return False


def processar_arquivos(arquivos_docx, chunk_size, chunk_overlap, embedding_model, collection):
    """
    Processa lista de arquivos DOCX
    """
    total_chunks_adicionados = 0

    for i, arquivo_path in enumerate(arquivos_docx, 1):
        
        try:
            texto = extrair_texto_docx(str(arquivo_path))
            if not texto:
                continue

            chunks = dividir_texto_em_chunks(texto, chunk_size, chunk_overlap)
            if not chunks:
                continue

            metadados = criar_metadados(chunks, arquivo_path, texto)

            # Gerar IDs únicos
            base_id = arquivo_path.stem  
            ids = [f"{base_id}_chunk_{j:03d}" for j in range(len(chunks))]
            
            # Verificar duplicatas
            if chunks_already_exists(collection, ids):
                continue
            
            # Criar embeddings
            embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
            
            # Adicionar ao ChromaDB
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadados,
                ids=ids
            )
            
            total_chunks_adicionados += len(chunks)
            
        except Exception as e:
            print(f"Erro ao processar '{arquivo_path.name}': {e}")
            continue
    
    return total_chunks_adicionados


def processar_documentos():
    """
    Função principal para processar todos os documentos DOCX
    """
    CHUNK_SIZE = 1000      
    CHUNK_OVERLAP = 200    
    
    # 1. Verificar pasta
    arquivos_docx = verificar_pasta_de_documentos()
    if not arquivos_docx:
        return False
    
    # 2. Configurar ChromaDB
    client = criar_chromadb(BASE_DADOS)
    if not client:
        return False

    collection = criar_colecao(client, COLECAO)
    if not collection:
        return False
    
    # 3. Carregar modelo
    embedding_model = carregar_modelo_embedding()
    if not embedding_model:
        return False
    
    # 4. Processar arquivos
    total_chunks_adicionados = processar_arquivos(
        arquivos_docx,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        embedding_model,
        collection
    )
    
    # 5. Resumo
    if total_chunks_adicionados == 0:
        print("\nErro: Nenhum documento foi processado com sucesso")
        print("Verifique se os arquivos .docx são válidos e não estão corrompidos")
        return False
    
    return True


def verificar_status():
    """
    Verifica o status atual da base de dados
    """
    print("\nSTATUS DA BASE DE DADOS")
    print("=" * 50)
    
    try:
        if not os.path.exists(BASE_DADOS):
            print("Erro: Base de dados não existe")
            print("Execute o processamento primeiro")
            return
        
        client = chromadb.PersistentClient(path=BASE_DADOS)
        
        try:
            collection = client.get_collection(COLECAO)
        except NotFoundError:
            print(f"Erro: Coleção '{COLECAO}' não existe")
            print("Execute o processamento para criar a coleção")
            return
        except Exception as e:
            print(f"Erro ao acessar coleção: {e}")
            return
        
        total = collection.count()
        print(f"Total de chunks: {total}")
        
        if total > 0:
            # Pegar alguns exemplos
            amostra = collection.get(limit=5)
            
            # Contar arquivos únicos
            arquivos = set()
            for metadata in amostra['metadatas']:
                arquivos.add(metadata.get('arquivo', 'Desconhecido'))
            
            print(f"Arquivos processados: {len(arquivos)}")
            for arquivo in sorted(arquivos):
                print(f"   - {arquivo}")
        else:
            print("Aviso: Base existe mas está vazia")
            
    except Exception as e:
        print(f"Erro ao verificar base: {e}")


def menu_principal():
    """
    Menu principal do script
    """
    while True:
        print("\n" + "=" * 50)
        print("PROCESSAMENTO DE DOCUMENTOS DOCX")
        print("=" * 50)
        print("1. Processar documentos DOCX")
        print("2. Ver status da base de dados")
        print("3. Limpar base de dados")
        print("4. Ajuda")
        print("5. Sair")
        print("-" * 50)
        
        escolha = input("Escolha uma opção (1-5): ").strip()
        
        if escolha == "1":
            sucesso = processar_documentos()
            if not sucesso:
                print("\nProcessamento falhou. Verifique os erros acima.")
                
        elif escolha == "2":
            verificar_status()
            
        elif escolha == "3":
            import shutil
            confirma = input("\nTem certeza que quer apagar a base? (sim/não): ")
            if confirma.lower() in ['sim', 's', 'yes', 'y']:
                try:
                    if os.path.exists(BASE_DADOS):
                        shutil.rmtree(BASE_DADOS)
                        print("Base de dados apagada")
                    else:
                        print("Base de dados não existe")
                except Exception as e:
                    print(f"Erro ao apagar: {e}")
            else:
                print("Operação cancelada")
                
        elif escolha == "4":
            print("\nAJUDA")
            print("-" * 50)
            print("1. Coloque arquivos .docx na pasta './documentos'")
            print("2. Execute a opção 1 para processar")
            print("3. Use a opção 2 para verificar o status")
            print("4. Depois configure o sistema RAG para fazer perguntas")
            print("\nDependências necessárias:")
            print("   pip install chromadb sentence-transformers python-docx")
            
        elif escolha == "5":
            print("\nAté logo!")
            break
            
        else:
            print("\nOpção inválida. Tente novamente.")


if __name__ == "__main__":
    # Verificar dependências básicas
    try:
        import chromadb
        import sentence_transformers
        import docx
    except ImportError as e:
        print("Erro: Dependência não encontrada!")
        print("Execute: pip install chromadb sentence-transformers python-docx")
        exit(1)
    
    # Executar menu principal
    menu_principal()