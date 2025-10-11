# rag/attachments.py - Attachment RAG with sentence transformers and FAISS

import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class AttachmentIndex:
    """
    RAG index for attachment documents using sentence transformers and FAISS.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = []
        self.texts = []
        self.metadata = []
        self.index = None
        self.encoder = None
        self.is_built = False
        
    def _load_encoder(self):
        """Load the sentence transformer model."""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence transformer model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                raise
    
    def _get_faiss_index(self, dimension: int):
        """Get or create FAISS index."""
        try:
            import faiss
            if self.index is None:
                # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
                self.index = faiss.IndexFlatIP(dimension)
                logger.info(f"Created FAISS index with dimension {dimension}")
            return self.index
        except ImportError:
            logger.warning("FAISS not available, using numpy-based similarity")
            return None
    
    def add_docs(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the index.
        
        Args:
            texts: List of text documents to add
            metadata: Optional metadata for each document
        """
        if not texts:
            return
        
        self._load_encoder()
        
        try:
            # Generate embeddings
            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to collections
            self.embeddings.extend(embeddings)
            self.texts.extend(texts)
            
            if metadata:
                self.metadata.extend(metadata)
            else:
                # Create default metadata
                for i, text in enumerate(texts):
                    self.metadata.append({
                        "id": len(self.metadata),
                        "text_length": len(text),
                        "source": "attachment"
                    })
            
            logger.info(f"Added {len(texts)} documents to index")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def build(self):
        """Build the search index."""
        if not self.embeddings:
            logger.warning("No embeddings to build index")
            return
        
        try:
            embeddings_array = np.array(self.embeddings).astype('float32')
            
            # Try FAISS first
            faiss_index = self._get_faiss_index(embeddings_array.shape[1])
            if faiss_index is not None:
                faiss_index.add(embeddings_array)
                self.index = faiss_index
                logger.info(f"Built FAISS index with {len(self.embeddings)} vectors")
            else:
                # Fallback to numpy-based search
                self.index = embeddings_array
                logger.info(f"Built numpy-based index with {len(self.embeddings)} vectors")
            
            self.is_built = True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise
    
    def query(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Query the index for similar documents.
        
        Args:
            query_text: Query text
            k: Number of results to return
        
        Returns:
            List of (text, score) tuples
        """
        if not self.is_built or not self.embeddings:
            logger.warning("Index not built or empty")
            return []
        
        try:
            self._load_encoder()
            
            # Generate query embedding
            query_embedding = self.encoder.encode([query_text], convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            if isinstance(self.index, np.ndarray):
                # Numpy-based similarity search
                similarities = np.dot(self.index, query_embedding.T).flatten()
                top_indices = np.argsort(similarities)[::-1][:k]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0:  # Only return positive similarities
                        results.append((self.texts[idx], float(similarities[idx])))
                
                return results
            
            else:
                # FAISS-based search
                scores, indices = self.index.search(query_embedding.astype('float32'), k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.texts) and score > 0:
                        results.append((self.texts[idx], float(score)))
                
                return results
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "num_documents": len(self.texts),
            "num_embeddings": len(self.embeddings),
            "is_built": self.is_built,
            "model_name": self.model_name,
            "index_type": "FAISS" if hasattr(self.index, 'search') else "numpy"
        }
    
    def save(self, path: str):
        """Save the index to disk."""
        try:
            save_data = {
                "embeddings": self.embeddings,
                "texts": self.texts,
                "metadata": self.metadata,
                "model_name": self.model_name,
                "is_built": self.is_built
            }
            
            with open(path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Saved index to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load(self, path: str):
        """Load the index from disk."""
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.embeddings = save_data["embeddings"]
            self.texts = save_data["texts"]
            self.metadata = save_data["metadata"]
            self.model_name = save_data["model_name"]
            self.is_built = save_data["is_built"]
            
            # Rebuild index if needed
            if self.is_built and self.embeddings:
                self.build()
            
            logger.info(f"Loaded index from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def clear(self):
        """Clear all data from the index."""
        self.embeddings = []
        self.texts = []
        self.metadata = []
        self.index = None
        self.is_built = False
        logger.info("Cleared index")

def create_attachment_index() -> AttachmentIndex:
    """Create a new attachment index."""
    return AttachmentIndex()

def process_attachment_file(file_path: str, index: AttachmentIndex) -> bool:
    """
    Process a single attachment file and add it to the index.
    
    Args:
        file_path: Path to the attachment file
        index: AttachmentIndex to add the content to
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from services.files import extract_text_from_file, split_chunks, clean_text
        from services.ocr import extract_text_from_image, is_image_file
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False
        
        # Extract text based on file type
        if is_image_file(file_path):
            text = extract_text_from_image(file_path)
        else:
            text = extract_text_from_file(file_path)
        
        if not text or text.startswith("Error") or text.strip().lower().startswith("no text content found") or len(text.strip()) < 50:
            logger.warning(f"No text extracted from {file_path}")
            return False
        
        # Clean and chunk the text
        clean_text_content = clean_text(text)
        chunks = split_chunks(clean_text_content, max_tokens=400, overlap=100)
        
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return False
        
        # Create metadata for each chunk
        file_name = os.path.basename(file_path)
        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata_list.append({
                "id": f"{file_name}_{i}",
                "source_file": file_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_type": "image" if is_image_file(file_path) else "document"
            })
        
        # Add to index
        index.add_docs(chunks, metadata_list)
        
        logger.info(f"Successfully processed {file_path} into {len(chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process attachment {file_path}: {e}")
        return False

def process_attachment_files(file_paths: List[str]) -> AttachmentIndex:
    """
    Process multiple attachment files and create an index.
    
    Args:
        file_paths: List of file paths to process
    
    Returns:
        AttachmentIndex with processed content
    """
    index = create_attachment_index()
    
    successful_files = []
    for file_path in file_paths:
        if process_attachment_file(file_path, index):
            successful_files.append(file_path)
    
    if successful_files:
        index.build()
        logger.info(f"Processed {len(successful_files)} files successfully")
    else:
        logger.warning("No files were processed successfully")
    
    return index
