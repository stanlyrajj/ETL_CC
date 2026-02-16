#!/usr/bin/env python3
"""
Part 2: FastAPI Backend with Processing & RAG Endpoints
Extends Part 1 with content extraction and teaching features
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from pathlib import Path
import logging
from datetime import datetime

# Import processing pipeline
from processor import (
    ProcessingPipeline, 
    ExtractedPaper,
    Chunk
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Paper System - Part 2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processing pipeline
pipeline = ProcessingPipeline()

# Global processing status
processing_status = {
    'is_processing': False,
    'current_paper': '',
    'progress': 0,
    'total': 0,
    'results': []
}


# ============================================================================
# MODELS
# ============================================================================

class ProcessRequest(BaseModel):
    paper_ids: Optional[List[str]] = None  # If None, process all
    source: str = Field(..., regex='^(arxiv|pubmed|all)$')


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict] = None  # e.g., {'paper_id': 'xxx', 'section_type': 'methods'}


class TeachingRequest(BaseModel):
    paper_id: str
    level: str = Field(..., regex='^(beginner|intermediate|advanced)$')


class CompareRequest(BaseModel):
    paper_ids: List[str] = Field(..., min_items=2, max_items=5)
    aspect: str  # 'methodology', 'results', 'applications'


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def process_papers_background(paper_paths: List[tuple]):
    """Background task to process multiple papers"""
    global processing_status
    
    processing_status['is_processing'] = True
    processing_status['total'] = len(paper_paths)
    processing_status['progress'] = 0
    processing_status['results'] = []
    
    for i, (file_path, source) in enumerate(paper_paths):
        processing_status['current_paper'] = Path(file_path).stem
        processing_status['progress'] = i + 1
        
        result = pipeline.process_paper(file_path, source)
        processing_status['results'].append(result)
        
        logger.info(f"Processed {i+1}/{len(paper_paths)}: {result['status']}")
    
    processing_status['is_processing'] = False
    logger.info("All papers processed")


async def generate_teaching_content_background(paper_id: str, level: str):
    """Background task to generate teaching content"""
    # This would call Claude API
    # For now, return placeholder
    logger.info(f"Generating {level} teaching content for {paper_id}")
    # Implementation would use Anthropic API here


# ============================================================================
# PROCESSING ENDPOINTS
# ============================================================================

@app.post("/api/process")
async def process_papers(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process downloaded papers: extract, chunk, embed, store
    """
    global processing_status
    
    if processing_status['is_processing']:
        raise HTTPException(
            status_code=409,
            detail="Processing already in progress"
        )
    
    # Collect files to process
    paper_paths = []
    
    if request.source in ['arxiv', 'all']:
        arxiv_dir = Path("downloads/arxiv")
        if arxiv_dir.exists():
            for pdf in arxiv_dir.glob("*.pdf"):
                paper_paths.append((str(pdf), 'arxiv'))
    
    if request.source in ['pubmed', 'all']:
        pubmed_dir = Path("downloads/pubmed")
        if pubmed_dir.exists():
            for json_file in pubmed_dir.glob("*.json"):
                paper_paths.append((str(json_file), 'pubmed'))
    
    # Filter by paper_ids if provided
    if request.paper_ids:
        paper_paths = [
            (path, src) for path, src in paper_paths
            if Path(path).stem in request.paper_ids
        ]
    
    if not paper_paths:
        raise HTTPException(
            status_code=404,
            detail="No papers found to process"
        )
    
    # Start background processing
    background_tasks.add_task(process_papers_background, paper_paths)
    
    return {
        'status': 'started',
        'message': f'Processing {len(paper_paths)} papers in background',
        'total_papers': len(paper_paths)
    }


@app.get("/api/process/status")
async def get_processing_status():
    """Get current processing status"""
    return processing_status


@app.post("/api/process/{paper_id}")
async def process_single_paper(paper_id: str, background_tasks: BackgroundTasks):
    """Process a specific paper"""
    # Find the paper file
    arxiv_path = Path(f"downloads/arxiv/{paper_id}.pdf")
    pubmed_path = Path(f"downloads/pubmed/{paper_id}.json")
    
    if arxiv_path.exists():
        file_path = str(arxiv_path)
        source = 'arxiv'
    elif pubmed_path.exists():
        file_path = str(pubmed_path)
        source = 'pubmed'
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Paper {paper_id} not found"
        )
    
    # Process immediately (not in background for single paper)
    result = pipeline.process_paper(file_path, source)
    
    return result


# ============================================================================
# RAG QUERY ENDPOINTS
# ============================================================================

@app.post("/api/query")
async def query_papers(request: QueryRequest):
    """
    Semantic search across all papers
    """
    try:
        results = pipeline.vector_store.query(
            query_text=request.query,
            n_results=request.top_k,
            filter_dict=request.filters
        )
        
        # Format response
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'chunk_id': results['ids'][i],
                'paper_id': results['metadatas'][i]['paper_id'],
                'section_type': results['metadatas'][i]['section_type'],
                'content': results['documents'][i],
                'relevance_score': 1 - results['distances'][i],  # Convert distance to similarity
                'metadata': results['metadatas'][i]
            })
        
        return {
            'query': request.query,
            'total_results': len(formatted_results),
            'results': formatted_results
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/paper/{paper_id}")
async def query_single_paper(paper_id: str, request: QueryRequest):
    """
    Search within a specific paper
    """
    # Add paper_id to filters
    filters = request.filters or {}
    filters['paper_id'] = paper_id
    
    request.filters = filters
    return await query_papers(request)


@app.post("/api/compare")
async def compare_papers(request: CompareRequest):
    """
    Compare multiple papers on a specific aspect
    """
    # Query each paper for the aspect
    comparisons = []
    
    for paper_id in request.paper_ids:
        query = f"{request.aspect} in this research"
        
        results = pipeline.vector_store.query(
            query_text=query,
            n_results=3,
            filter_dict={'paper_id': paper_id}
        )
        
        if results['documents']:
            comparisons.append({
                'paper_id': paper_id,
                'relevant_sections': [
                    {
                        'content': doc,
                        'section': meta['section_type']
                    }
                    for doc, meta in zip(results['documents'], results['metadatas'])
                ]
            })
    
    return {
        'aspect': request.aspect,
        'papers_compared': len(comparisons),
        'comparisons': comparisons
    }


# ============================================================================
# TEACHING ENDPOINTS
# ============================================================================

@app.get("/api/teach/{paper_id}/{level}")
async def get_teaching_content(paper_id: str, level: str):
    """
    Get teaching content for a paper at specified level
    
    In production, this would:
    1. Check if content exists in database
    2. If not, generate it using Claude API
    3. Cache the result
    """
    
    # Placeholder response (would come from database in production)
    teaching_templates = {
        'beginner': {
            'summary': f"This paper explains the research in simple terms...",
            'analogies': [
                "Think of this like...",
                "Imagine if..."
            ],
            'key_terms': {
                'term1': 'Simple definition',
                'term2': 'Easy to understand explanation'
            }
        },
        'intermediate': {
            'summary': f"Technical overview with context...",
            'methodology_breakdown': [
                "Step 1: Data collection",
                "Step 2: Analysis approach"
            ],
            'key_insights': [
                "Novel contribution 1",
                "Important finding 2"
            ]
        },
        'advanced': {
            'summary': f"Detailed technical analysis...",
            'critique': {
                'strengths': ["Rigorous methodology", "Novel approach"],
                'weaknesses': ["Limited sample size", "Scope constraints"]
            },
            'applications': [
                "Potential use case 1",
                "Future research direction"
            ]
        }
    }
    
    if level not in teaching_templates:
        raise HTTPException(status_code=400, detail="Invalid level")
    
    return {
        'paper_id': paper_id,
        'level': level,
        'content': teaching_templates[level],
        'generated_at': datetime.now().isoformat()
    }


@app.post("/api/teach/regenerate")
async def regenerate_teaching(request: TeachingRequest, background_tasks: BackgroundTasks):
    """
    Regenerate teaching content for a paper
    """
    background_tasks.add_task(
        generate_teaching_content_background,
        request.paper_id,
        request.level
    )
    
    return {
        'status': 'started',
        'message': f'Regenerating {request.level} content for {request.paper_id}'
    }


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/api/papers")
async def list_papers(
    source: Optional[str] = None,
    processed_only: bool = False
):
    """
    List all papers with metadata
    """
    papers = []
    
    # In production, this would query the database
    # For now, list files
    
    if not source or source == 'arxiv':
        arxiv_dir = Path("downloads/arxiv")
        if arxiv_dir.exists():
            for pdf in arxiv_dir.glob("*.pdf"):
                papers.append({
                    'paper_id': pdf.stem,
                    'source': 'arxiv',
                    'local_path': str(pdf),
                    'file_size': pdf.stat().st_size
                })
    
    if not source or source == 'pubmed':
        pubmed_dir = Path("downloads/pubmed")
        if pubmed_dir.exists():
            for json_file in pubmed_dir.glob("*.json"):
                papers.append({
                    'paper_id': json_file.stem,
                    'source': 'pubmed',
                    'local_path': str(json_file),
                    'file_size': json_file.stat().st_size
                })
    
    return {
        'total': len(papers),
        'papers': papers
    }


@app.get("/api/papers/{paper_id}/stats")
async def get_paper_stats(paper_id: str):
    """
    Get statistics for a processed paper
    """
    # Query vector store for chunks
    results = pipeline.vector_store.collection.get(
        where={'paper_id': paper_id}
    )
    
    if not results['ids']:
        raise HTTPException(
            status_code=404,
            detail="Paper not found or not processed"
        )
    
    # Calculate stats
    total_chunks = len(results['ids'])
    sections = set(meta['section_type'] for meta in results['metadatas'])
    
    return {
        'paper_id': paper_id,
        'total_chunks': total_chunks,
        'sections': list(sections),
        'section_count': len(sections),
        'processed': True
    }


@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """
    Delete a paper and all associated data
    """
    try:
        # Delete from vector store
        pipeline.vector_store.delete_paper(paper_id)
        
        # Delete files
        arxiv_path = Path(f"downloads/arxiv/{paper_id}.pdf")
        pubmed_path = Path(f"downloads/pubmed/{paper_id}.json")
        
        deleted_files = []
        if arxiv_path.exists():
            arxiv_path.unlink()
            deleted_files.append(str(arxiv_path))
        
        if pubmed_path.exists():
            pubmed_path.unlink()
            deleted_files.append(str(pubmed_path))
        
        return {
            'status': 'deleted',
            'paper_id': paper_id,
            'deleted_files': deleted_files
        }
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API information"""
    return {
        'title': 'Research Paper System - Part 2',
        'version': '2.0.0',
        'features': [
            'Content extraction (PDF & JSON)',
            'Text chunking & embedding',
            'Vector search (RAG)',
            'Teaching content generation',
            'Paper comparison'
        ],
        'endpoints': {
            'processing': ['/api/process', '/api/process/status'],
            'rag': ['/api/query', '/api/query/paper/{id}', '/api/compare'],
            'teaching': ['/api/teach/{id}/{level}', '/api/teach/regenerate'],
            'utility': ['/api/papers', '/api/papers/{id}/stats']
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)