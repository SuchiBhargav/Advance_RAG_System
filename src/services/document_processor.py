"""
Document processing service for handling file uploads and incremental indexing.
Supports multiple file formats with proper chunking and metadata extraction.
"""

from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import io
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)

from config.settings import settings
from src.utils.logger import get_logger
from src.core.vector_store import HybridVectorStore

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Processes documents for indexing with proper chunking and metadata.
    Supports incremental updates and multiple file formats.
    """
    
    def __init__(self, vector_store: HybridVectorStore):
        """
        Initialize document processor.
        
        Args:
            vector_store: Vector store instance for indexing
        """
        self.vector_store = vector_store
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len
        )
        
        logger.info("Initialized DocumentProcessor")
    
    async def process_document(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document and add it to the vector store.
        
        Args:
            content: Document content as bytes
            filename: Name of the file
            metadata: Additional metadata
        
        Returns:
            Processing result with document ID and chunk count
        """
        try:
            logger.info(f"Processing document: {filename}")
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Extract text based on file type
            file_ext = Path(filename).suffix.lower()
            text = await self._extract_text(content, file_ext, filename)
            
            if not text or not text.strip():
                raise ValueError("No text content extracted from document")
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Prepare metadata for each chunk
            base_metadata = {
                "source": filename,
                "document_id": document_id,
                "file_type": file_ext.lstrip('.'),
                "created_at": datetime.utcnow().isoformat(),
                "total_chunks": len(chunks)
            }
            
            # Add custom metadata
            if metadata:
                base_metadata.update(metadata)
            
            # Create chunk metadata
            chunk_metadatas = []
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                # Generate a valid UUID for Qdrant (no underscores or special chars)
                chunk_id = str(uuid.uuid4())
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_id": chunk_id
                })
                
                chunk_metadatas.append(chunk_metadata)
                chunk_ids.append(chunk_id)
            
            # Add to vector store
            self.vector_store.add_documents(
                texts=chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            logger.info(
                f"Successfully processed document {filename}: "
                f"{len(chunks)} chunks indexed"
            )
            
            return {
                "document_id": document_id,
                "chunks_created": len(chunks),
                "filename": filename,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise
    
    async def _extract_text(
        self,
        content: bytes,
        file_ext: str,
        filename: str
    ) -> str:
        """
        Extract text from document based on file type.
        
        Args:
            content: Document content
            file_ext: File extension
            filename: File name
        
        Returns:
            Extracted text
        """
        try:
            if file_ext == '.pdf':
                return await self._extract_from_pdf(content, filename)
            elif file_ext == '.txt':
                return content.decode('utf-8')
            elif file_ext == '.md':
                return content.decode('utf-8')
            elif file_ext in ['.doc', '.docx']:
                return await self._extract_from_docx(content, filename)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            raise
    
    async def _extract_from_pdf(self, content: bytes, filename: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            content: PDF content
            filename: File name
        
        Returns:
            Extracted text
        """
        try:
            # Save temporarily to process with PyPDFLoader
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            # Load and extract
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Combine all pages
            text = "\n\n".join([doc.page_content for doc in documents])
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            raise
    
    async def _extract_from_docx(self, content: bytes, filename: str) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            content: DOCX content
            filename: File name
        
        Returns:
            Extracted text
        """
        try:
            # Save temporarily
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            # Load and extract
            loader = UnstructuredWordDocumentLoader(temp_path)
            documents = loader.load()
            
            text = "\n\n".join([doc.page_content for doc in documents])
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {e}")
            raise
    
    async def update_document(
        self,
        document_id: str,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update an existing document (incremental indexing).
        
        Args:
            document_id: ID of document to update
            content: New document content
            filename: File name
            metadata: Updated metadata
        
        Returns:
            Update result
        """
        try:
            logger.info(f"Updating document: {document_id}")
            
            # Delete old chunks
            # Note: This is a simple implementation. For true incremental updates,
            # you'd want to compare chunks and only update changed ones
            old_chunk_ids = [
                f"{document_id}_chunk_{i}" for i in range(1000)  # Assume max 1000 chunks
            ]
            self.vector_store.delete_documents(old_chunk_ids)
            
            # Process as new document with same ID
            result = await self.process_document(content, filename, metadata)
            result["document_id"] = document_id
            result["status"] = "updated"
            
            logger.info(f"Successfully updated document {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            raise
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """
        Get information about a document.
        
        Args:
            document_id: Document ID
        
        Returns:
            Document information
        """
        # This would query the vector store for document metadata
        # Implementation depends on vector store capabilities
        return {
            "document_id": document_id,
            "status": "exists"
        }

# Made with Bob
