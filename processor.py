#!/usr/bin/env python3
"""
Part 2: Research Paper Processing System
Extracts, processes, and prepares papers for RAG and teaching
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

import fitz  # PyMuPDF
import ftfy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Section:
    """Represents a section of a paper"""
    type: str  # 'abstract', 'introduction', 'methods', etc.
    title: Optional[str]
    content: str
    order_index: int
    word_count: int


@dataclass
class ExtractedPaper:
    """Complete extracted paper content"""
    paper_id: str
    source: str
    title: str
    authors: List[str]
    abstract: str
    sections: List[Section]
    full_text: str
    metadata: Dict


@dataclass
class Chunk:
    """Text chunk for RAG"""
    chunk_id: str
    paper_id: str
    section_type: str
    chunk_text: str
    chunk_index: int
    token_count: int
    metadata: Dict


# ============================================================================
# PDF EXTRACTION (arXiv)
# ============================================================================

class PDFExtractor:
    """Extracts content from arXiv PDFs"""
    
    SECTION_PATTERNS = [
        r'^(?:abstract|introduction|background|related work|methodology?|methods|'
        r'approach|implementation|experiments?|results?|discussion|conclusion|'
        r'future work|references|acknowledgments?)\s*$'
    ]
    
    def __init__(self):
        self.section_regex = re.compile('|'.join(self.SECTION_PATTERNS), re.IGNORECASE)
    
    def extract(self, pdf_path: str) -> ExtractedPaper:
        """Extract full content from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            paper_id = Path(pdf_path).stem
            title = self._extract_title(doc)
            authors = self._extract_authors(doc)
            
            # Extract full text
            full_text = self._extract_full_text(doc)
            
            # Clean text
            full_text = self._clean_text(full_text)
            
            # Detect sections
            sections = self._detect_sections(full_text)
            
            # Extract abstract separately
            abstract = self._find_abstract(sections)
            
            doc.close()
            
            return ExtractedPaper(
                paper_id=paper_id,
                source='arxiv',
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                full_text=full_text,
                metadata={
                    'extraction_date': datetime.now().isoformat(),
                    'total_sections': len(sections),
                    'word_count': len(full_text.split())
                }
            )
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            raise
    
    def _extract_title(self, doc) -> str:
        """Extract title from first page"""
        first_page = doc[0].get_text()
        lines = [l.strip() for l in first_page.split('\n') if l.strip()]
        
        # Title is usually the first substantial line
        for line in lines[:10]:
            if len(line) > 20 and not line.isupper():
                return line
        
        return "Unknown Title"
    
    def _extract_authors(self, doc) -> List[str]:
        """Extract authors from first page"""
        first_page = doc[0].get_text()
        
        # Simple heuristic: lines after title, before abstract
        lines = [l.strip() for l in first_page.split('\n') if l.strip()]
        authors = []
        
        for i, line in enumerate(lines[1:10]):
            if 'abstract' in line.lower():
                break
            if '@' in line or 'university' in line.lower():
                continue
            if len(line) < 50 and ',' in line:
                authors.extend([a.strip() for a in line.split(',')])
        
        return authors[:5] if authors else ["Unknown Authors"]
    
    def _extract_full_text(self, doc) -> str:
        """Extract all text from PDF"""
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        return '\n\n'.join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean PDF artifacts and normalize text"""
        # Fix encoding issues
        text = ftfy.fix_text(text)
        
        # Fix common ligatures
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff').replace('ﬃ', 'ffi')
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove page numbers (heuristic)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        return text.strip()
    
    def _detect_sections(self, text: str) -> List[Section]:
        """Detect and extract sections"""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        order_index = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if line is a section header
            if self.section_regex.match(line_stripped):
                # Save previous section
                if current_section:
                    content = '\n'.join(current_content).strip()
                    sections.append(Section(
                        type=current_section.lower().replace(' ', '_'),
                        title=current_section,
                        content=content,
                        order_index=order_index,
                        word_count=len(content.split())
                    ))
                    order_index += 1
                
                # Start new section
                current_section = line_stripped
                current_content = []
            
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            sections.append(Section(
                type=current_section.lower().replace(' ', '_'),
                title=current_section,
                content=content,
                order_index=order_index,
                word_count=len(content.split())
            ))
        
        return sections
    
    def _find_abstract(self, sections: List[Section]) -> str:
        """Extract abstract from sections"""
        for section in sections:
            if section.type == 'abstract':
                return section.content
        return ""


# ============================================================================
# JSON EXTRACTION (PubMed)
# ============================================================================

class PubMedExtractor:
    """Extracts content from PubMed BioC JSON"""
    
    def extract(self, json_path: str) -> ExtractedPaper:
        """Extract content from BioC JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            paper_id = Path(json_path).stem
            
            # Navigate BioC structure
            documents = data.get('documents', [])
            if not documents:
                raise ValueError("No documents in BioC JSON")
            
            doc = documents[0]
            passages = doc.get('passages', [])
            
            # Extract metadata
            title = self._extract_title(passages)
            authors = []  # Not in BioC passages
            
            # Group passages into sections
            sections = self._group_into_sections(passages)
            
            # Extract abstract
            abstract = self._find_abstract(sections)
            
            # Build full text
            full_text = '\n\n'.join([s.content for s in sections])
            
            return ExtractedPaper(
                paper_id=paper_id,
                source='pubmed',
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                full_text=full_text,
                metadata={
                    'extraction_date': datetime.now().isoformat(),
                    'total_sections': len(sections),
                    'word_count': len(full_text.split())
                }
            )
            
        except Exception as e:
            logger.error(f"JSON extraction failed for {json_path}: {e}")
            raise
    
    def _extract_title(self, passages: List[Dict]) -> str:
        """Extract title from passages"""
        for passage in passages:
            infons = passage.get('infons', {})
            if infons.get('type') == 'title' or infons.get('section_type') == 'TITLE':
                return passage.get('text', '').strip()
        return "Unknown Title"
    
    def _group_into_sections(self, passages: List[Dict]) -> List[Section]:
        """Group passages into logical sections"""
        sections = []
        order_index = 0
        
        for passage in passages:
            infons = passage.get('infons', {})
            section_type = infons.get('section_type', infons.get('type', 'body'))
            text = passage.get('text', '').strip()
            
            if not text:
                continue
            
            # Normalize section type
            section_type = section_type.lower()
            
            sections.append(Section(
                type=section_type,
                title=section_type.title(),
                content=text,
                order_index=order_index,
                word_count=len(text.split())
            ))
            order_index += 1
        
        return sections
    
    def _find_abstract(self, sections: List[Section]) -> str:
        """Extract abstract"""
        for section in sections:
            if section.type in ['abstract', 'abs']:
                return section.content
        return ""


# ============================================================================
# TEXT CHUNKING
# ============================================================================

class TextChunker:
    """Chunks text for RAG embeddings"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_sections(self, paper_id: str, sections: List[Section]) -> List[Chunk]:
        """Chunk all sections of a paper"""
        all_chunks = []
        
        for section in sections:
            section_chunks = self._chunk_text(section.content)
            
            for i, chunk_text in enumerate(section_chunks):
                chunk = Chunk(
                    chunk_id=f"{paper_id}_sec{section.order_index}_chunk{i}",
                    paper_id=paper_id,
                    section_type=section.type,
                    chunk_text=chunk_text,
                    chunk_index=i,
                    token_count=len(chunk_text.split()),
                    metadata={
                        'section_title': section.title,
                        'section_order': section.order_index,
                        'total_chunks_in_section': len(section_chunks),
                        'chunk_position': f"{i+1}/{len(section_chunks)}"
                    }
                )
                all_chunks.append(chunk)
        
        return all_chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period followed by space
                last_period = text.rfind('. ', start, end)
                if last_period > start:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position (with overlap)
            start = end - self.chunk_overlap
        
        return chunks


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

class EmbeddingGenerator:
    """Generates embeddings for text chunks"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def generate_for_chunks(self, chunks: List[Chunk]) -> List[Tuple[str, List[float]]]:
        """Generate embeddings with chunk IDs"""
        texts = [c.chunk_text for c in chunks]
        embeddings = self.generate(texts)
        return [(c.chunk_id, emb) for c, emb in zip(chunks, embeddings)]


# ============================================================================
# VECTOR STORE (ChromaDB)
# ============================================================================

class VectorStore:
    """Manages ChromaDB vector store"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="paper_chunks",
            metadata={"description": "Research paper chunks for RAG"}
        )
        
        logger.info(f"ChromaDB initialized: {self.collection.count()} existing chunks")
    
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Add chunks with embeddings to vector store"""
        ids = [c.chunk_id for c in chunks]
        documents = [c.chunk_text for c in chunks]
        metadatas = [
            {
                'paper_id': c.paper_id,
                'section_type': c.section_type,
                'chunk_index': c.chunk_index,
                **c.metadata
            }
            for c in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def query(self, query_text: str, n_results: int = 5, 
              filter_dict: Optional[Dict] = None) -> Dict:
        """Query the vector store"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_dict
        )
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def delete_paper(self, paper_id: str):
        """Delete all chunks for a paper"""
        self.collection.delete(where={'paper_id': paper_id})
        logger.info(f"Deleted chunks for paper: {paper_id}")


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

class ProcessingPipeline:
    """Orchestrates the entire processing workflow"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.json_extractor = PubMedExtractor()
        self.chunker = TextChunker()
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = VectorStore()
    
    def process_paper(self, file_path: str, source: str) -> Dict:
        """Process a single paper through entire pipeline"""
        logger.info(f"Processing {source} paper: {file_path}")
        
        try:
            # Stage 1: Extract content
            if source == 'arxiv':
                extracted = self.pdf_extractor.extract(file_path)
            else:
                extracted = self.json_extractor.extract(file_path)
            
            logger.info(f"Extracted {len(extracted.sections)} sections")
            
            # Stage 2: Chunk text
            chunks = self.chunker.chunk_sections(
                extracted.paper_id, 
                extracted.sections
            )
            logger.info(f"Created {len(chunks)} chunks")
            
            # Stage 3: Generate embeddings
            chunk_embeddings = self.embedding_gen.generate_for_chunks(chunks)
            embeddings = [emb for _, emb in chunk_embeddings]
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Stage 4: Store in vector DB
            self.vector_store.add_chunks(chunks, embeddings)
            
            return {
                'status': 'success',
                'paper_id': extracted.paper_id,
                'title': extracted.title,
                'sections': len(extracted.sections),
                'chunks': len(chunks),
                'word_count': extracted.metadata['word_count']
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def process_directory(self, directory: Path, source: str):
        """Process all papers in a directory"""
        if source == 'arxiv':
            pattern = "*.pdf"
        else:
            pattern = "*.json"
        
        files = list(directory.glob(pattern))
        logger.info(f"Found {len(files)} {source} papers to process")
        
        results = []
        for file_path in files:
            result = self.process_paper(str(file_path), source)
            results.append(result)
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    pipeline = ProcessingPipeline()
    
    # Process arXiv papers
    arxiv_results = pipeline.process_directory(
        Path("downloads/arxiv"), 
        source='arxiv'
    )
    
    # Process PubMed papers
    pubmed_results = pipeline.process_directory(
        Path("downloads/pubmed"), 
        source='pubmed'
    )
    
    # Example query
    query_results = pipeline.vector_store.query(
        query_text="How does CRISPR target specific genes?",
        n_results=5,
        filter_dict={'section_type': 'methods'}
    )
    
    print("\n=== Query Results ===")
    for i, (doc, meta) in enumerate(zip(
        query_results['documents'], 
        query_results['metadatas']
    )):
        print(f"\n{i+1}. Paper: {meta['paper_id']}")
        print(f"   Section: {meta['section_type']}")
        print(f"   Text: {doc[:200]}...")