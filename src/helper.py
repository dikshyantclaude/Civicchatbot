"""Small compatibility helpers for PDF ingestion; model imports are lazy."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Document:
    page_content: str
    metadata: dict

def extract_data_from_pdf_directory(directory):
    from pypdf import PdfReader
    docs=[]
    for path in sorted(Path(directory).glob('*.pdf')):
        for page_number,page in enumerate(PdfReader(path).pages):
            text=page.extract_text() or ''
            if text.strip():docs.append(Document(text,{'source':path.name,'page':page_number}))
    return docs

def optimized_text_splitter(documents):
    return [Document(doc.page_content[i:i+500],dict(doc.metadata)) for doc in documents for i in range(0,len(doc.page_content),480)]

def download_embeddings():
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',cache_folder=str(Path(__file__).resolve().parents[1]/'instance'/'models'))
    class Embeddings:
        def embed_documents(self,texts):return model.encode(texts,normalize_embeddings=True).tolist()
        def embed_query(self,text):return model.encode(text,normalize_embeddings=True).tolist()
    return Embeddings()
