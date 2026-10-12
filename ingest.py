"""Explicitly ingest PDF text into an existing 384-dimensional Pinecone index."""
import argparse
import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv
from src.helper import extract_data_from_pdf_directory, optimized_text_splitter, download_embeddings

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    if not args.directory.is_dir():parser.error('PDF directory does not exist')
    load_dotenv(Path(__file__).resolve().parent/'.env')
    if not os.getenv('PINECONE_API_KEY'):parser.error('Set PINECONE_API_KEY in .env')
    from pinecone import Pinecone
    index=Pinecone(api_key=os.environ['PINECONE_API_KEY']).Index(os.getenv('PINECONE_INDEX','chatbot'))
    stats=index.describe_index_stats()
    if stats['dimension']!=384:parser.error('Index must have dimension 384 for all-MiniLM-L6-v2')
    chunks=optimized_text_splitter(extract_data_from_pdf_directory(args.directory))
    if not chunks:parser.error('No readable PDF text found')
    embeddings=download_embeddings()
    for start in range(0,len(chunks),64):
        batch=chunks[start:start+64]
        values=embeddings.embed_documents([d.page_content for d in batch])
        vectors=[]
        for doc,vector in zip(batch,values):
            identity=doc.metadata['source']+str(doc.metadata['page'])+doc.page_content
            vectors.append({'id':hashlib.sha256(identity.encode()).hexdigest(),'values':vector,'metadata':dict(doc.metadata,text=doc.page_content)})
        index.upsert(vectors=vectors,namespace=os.getenv('PINECONE_NAMESPACE','default'))
    print(f'Indexed {len(chunks)} text chunks. Re-running unchanged PDFs reuses their IDs.')
if __name__=='__main__':main()
