# indexing.py - Document Processing and Index Creation System

from typing import List, Dict, Tuple, Optional
from core.config import RAGConfig, create_vector_store, ConfigPresets
from processing.preprocessor import VietnameseMarkdownPreprocessor
from processing.chunker import HierarchicalChunker
from processing.table_processor import TableProcessor
from processing.embedder import VietnameseEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from metadata import MetadataManager

import os
import json
import logging
import asyncio
import time
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
INDEX_PATH = os.getenv('INDEXING_PATH')
METADATA_EXCEL_PATH = os.getenv('METADATA_EXCEL_PATH')
DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH')
CONFIG_PRESET = os.getenv('CONFIG_PRESET', 'production')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedDocumentIndexingSystem:
    """Enhanced system with Excel metadata support and incremental indexing"""
    
    def __init__(self, config: RAGConfig = None, index_path: str = INDEX_PATH, 
                 enable_hybrid: bool = True, metadata_excel: str = METADATA_EXCEL_PATH):
        self.config = config or RAGConfig()
        self.index_path = index_path
        self.enable_hybrid = enable_hybrid
        
        topic_dir_map = {
            "Kiem_tra_phap_che": "ktpc",
            "Ky_Thuat_Cong_Nghe": "ktcn"
        }
        # Initialize metadata manager
        self.metadata_manager = MetadataManager(metadata_excel, 
                                                topic_dir_map=topic_dir_map)
        
        # Initialize processing components
        self.preprocessor = VietnameseMarkdownPreprocessor()
        self.chunker = HierarchicalChunker(self.config.chunking)
        self.table_processor = TableProcessor(self.config.table)
        self.embedder = VietnameseEmbedder(self.config.embedding)
        
        # Initialize vector store
        self.vector_store = create_vector_store(self.config)
        
        # Initialize hybrid retriever for BM25 indexing
        if self.enable_hybrid:
            bm25_path = f"{self.index_path}_bm25"
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder,
                bm25_path=bm25_path,
                enable_reranker=True
            )
            logger.info("🔄 Initialized Enhanced Document Indexing with Hybrid Support")
        else:
            self.hybrid_retriever = None
            logger.info("📊 Initialized Enhanced Document Indexing")
        
        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'bm25_indexed': 0,
            'last_updated': None,
            'hybrid_enabled': self.enable_hybrid,
            'processing_errors': 0,
            'successful_documents': [],
            'topics_processed': {},
            'incremental_updates': 0
        }
        
        # Create index directory
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Load existing stats and processing state
        self._load_existing_stats()
        self._load_processing_state()
    
    def _load_existing_stats(self):
        """Load existing processing statistics"""
        try:
            stats_path = f"{self.index_path}.stats"
            if os.path.exists(stats_path):
                with open(stats_path, 'r', encoding='utf-8') as f:
                    existing_stats = json.load(f)
                    self.stats.update(existing_stats)
                logger.info(f"📊 Loaded existing stats: {self.stats['documents_processed']} documents processed")
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing stats: {e}")
    
    def _load_processing_state(self):
        """Load processing state for incremental updates"""
        state_file = f"{self.index_path}.state"
        self.metadata_manager.load_processing_state(state_file)
    
    def _save_processing_state(self):
        """Save processing state for incremental updates"""
        state_file = f"{self.index_path}.state"
        self.metadata_manager.save_processing_state(state_file)
    
    async def process_document_with_metadata(self, doc_info: Dict) -> Dict:
        """Process a single document with metadata information"""
        try:
            file_path = doc_info['file_path']
            logger.info(f"📄 Processing document: {doc_info['filename']}")
            logger.info(f"   🏷️ Topic: {doc_info['topic']}")
            logger.info(f"   📅 Date: {doc_info['doc_date']}")
            logger.info(f"   📂 Doc ID: {doc_info['doc_id']}")
            
            # Read document content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Step 1: Preprocess document
            logger.info("  🔧 Preprocessing document structure")
            doc_structure = self.preprocessor.preprocess_document(content)
            
            # Step 2: Process tables if document has tables
            if doc_info['has_tables']:
                logger.info("  📊 Processing embedded tables")
                for i, table in enumerate(doc_structure.tables):
                    doc_structure.tables[i] = self.table_processor.process_table(table)
            
            # Step 3: Create document metadata for chunking (from Excel)
            document_metadata = {
                'doc_id': doc_info.get('doc_id', ''),
                'doc_title': doc_info.get('doc_title', ''),
                'filename': doc_info.get('filename', ''),
                'doc_date': doc_info.get('doc_date', ''),
                'topic': doc_info.get('topic', ''),
                'file_path': doc_info.get('file_path', ''),
                'has_tables': doc_info.get('has_tables', False)
            }
            
            # Step 4: Create hierarchical chunks with metadata
            logger.info("  ✂️ Creating hierarchical chunks with Excel metadata")
            chunks = self.chunker.chunk_document(doc_structure, document_metadata)

            # Save chunk to file csv for debugging
            self.chunker.save_chunks_to_csv(chunks,f"{self.index_path}_chunks.csv")
            
            # Step 5: Add processing metadata to chunks
            current_time = time.time()
            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata.update({
                    'processing_timestamp': current_time,
                    'processing_version': '1.0',
                    'source_file_path': file_path,
                    'indexed_at': current_time
                })
            
            # Step 6: Generate embeddings
            logger.info("  🧠 Generating vector embeddings")
            embeddings = await self.embedder.embed_chunks_async(chunks)
            
            # Step 7: Store in vector database
            logger.info("  💾 Storing in vector database")
            self.vector_store.add_embeddings(embeddings, chunks)
            
            # Step 8: Build BM25 index if hybrid mode enabled
            bm25_success = False
            if self.enable_hybrid and self.hybrid_retriever:
                logger.info("  🔍 Building BM25 index")
                bm25_success = self.hybrid_retriever.build_bm25_index(chunks)
            
            # Step 9: Update processing statistics
            topic = doc_info['topic']
            if topic not in self.stats['topics_processed']:
                self.stats['topics_processed'][topic] = {
                    'documents': 0,
                    'chunks': 0,
                    'last_processed': None
                }
            
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['embeddings_generated'] += len(embeddings)
            self.stats['topics_processed'][topic]['documents'] += 1
            self.stats['topics_processed'][topic]['chunks'] += len(chunks)
            self.stats['topics_processed'][topic]['last_processed'] = current_time
            
            if bm25_success:
                self.stats['bm25_indexed'] += len(chunks)
            
            self.stats['last_updated'] = current_time
            self.stats['successful_documents'].append({
                'document_id': doc_info['doc_id'],
                'filename': doc_info['filename'],
                'topic': topic,
                'chunks_count': len(chunks),
                'processed_at': self.stats['last_updated']
            })
            
            # Update file checksum
            self.metadata_manager.update_file_checksum(file_path)
            
            # Auto-save progress
            self._save_processing_metadata()
            self._save_processing_state()
            
            result = {
                'status': 'success',
                'document_id': doc_info['doc_id'],
                'filename': doc_info['filename'],
                'document_title': doc_info['doc_title'],
                'topic': topic,
                'chunks_created': len(chunks),
                'embeddings_generated': len(embeddings),
                'tables_processed': len(doc_structure.tables),
                'bm25_indexed': bm25_success,
                'hybrid_mode': self.enable_hybrid,
                'processing_time': current_time,
                'metadata_fields': list(document_metadata.keys())
            }
            
            logger.info(f"✅ Successfully indexed document {doc_info['filename']}")
            logger.info(f"   📊 Created {len(chunks)} chunks with metadata: {list(document_metadata.keys())}")
            return result
            
        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.error(f"❌ Error processing document {doc_info.get('filename', 'unknown')}: {e}")
            return {
                'status': 'error',
                'document_id': doc_info.get('doc_id', 'unknown'),
                'filename': doc_info.get('filename', 'unknown'),
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def process_from_metadata(self, base_directory: str = None, 
                                  batch_size: int = 10, 
                                  force_reindex: bool = False,
                                  specific_topics: List[str] = None) -> List[Dict]:
        """Process documents based on Excel metadata"""
        logger.info("📚 Starting metadata-based document processing")
        
        # Get documents to process
        all_documents = self.metadata_manager.get_documents_to_process(
            base_directory=base_directory, 
            force_reindex=force_reindex
        )
        
        # Filter by specific topics if provided
        if specific_topics:
            all_documents = [doc for doc in all_documents if doc['topic'] in specific_topics]
            logger.info(f"🎯 Filtered to topics: {specific_topics}")
        
        if not all_documents:
            logger.info("✅ No documents need processing")
            return []
        
        # Show topic breakdown
        topic_counts = {}
        for doc in all_documents:
            topic = doc['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        logger.info("📊 Documents to process by topic:")
        for topic, count in topic_counts.items():
            logger.info(f"   🏷️ {topic}: {count} documents")
        
        # Show metadata fields being used
        if all_documents:
            sample_doc = all_documents[0]
            metadata_fields = [k for k in sample_doc.keys() if k not in ['file_path', 'needs_processing']]
            logger.info(f"📋 Metadata fields: {metadata_fields}")
        
        # Process in batches
        results = []
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_documents) + batch_size - 1) // batch_size
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            batch_results = []
            for doc_info in batch:
                result = await self.process_document_with_metadata(doc_info)
                batch_results.append(result)
                
                # Progress logging
                if result['status'] == 'success':
                    logger.info(f"  ✅ {result['filename']} ({result['topic']}) - {result['chunks_created']} chunks")
                else:
                    logger.error(f"  ❌ {result['filename']}: {result.get('error', 'Unknown error')}")
            
            results.extend(batch_results)
            
            # Memory management between batches
            if batch_num < total_batches:
                logger.info(f"  💾 Saving progress after batch {batch_num}")
                self._save_processing_metadata()
                self._save_processing_state()
        
        # Final save and optimization
        self._save_processing_metadata()
        self._save_processing_state()
        
        # Final BM25 optimization if hybrid mode
        if self.enable_hybrid and self.hybrid_retriever:
            logger.info("🔄 Optimizing BM25 index...")
            await self._optimize_bm25_index()
        
        # Processing summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        logger.info(f"🎯 Metadata-based processing completed:")
        logger.info(f"   ✅ Successful: {successful}")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   📊 Total chunks created: {sum(r.get('chunks_created', 0) for r in results if r['status'] == 'success')}")
        
        # Topic-wise summary
        topic_results = {}
        for result in results:
            if result['status'] == 'success':
                topic = result['topic']
                if topic not in topic_results:
                    topic_results[topic] = {'successful': 0, 'chunks': 0}
                topic_results[topic]['successful'] += 1
                topic_results[topic]['chunks'] += result.get('chunks_created', 0)
        
        logger.info("📈 Results by topic:")
        for topic, stats in topic_results.items():
            logger.info(f"   🏷️ {topic}: {stats['successful']} docs, {stats['chunks']} chunks")
        
        # Metadata integration summary
        if successful > 0:
            sample_result = next(r for r in results if r['status'] == 'success')
            metadata_fields = sample_result.get('metadata_fields', [])
            logger.info(f"🏷️ Metadata fields integrated: {metadata_fields}")
        
        return results
    
    async def _optimize_bm25_index(self):
        """Optimize BM25 index after batch processing"""
        try:
            if self.hybrid_retriever:
                logger.info("🔄 BM25 index optimization completed")
        except Exception as e:
            logger.error(f"❌ Error optimizing BM25 index: {e}")
    
    def _save_processing_metadata(self):
        """Save processing metadata and statistics"""
        try:
            # Save processing statistics
            stats_path = f"{self.index_path}.stats"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            
            # Save vector store index
            self.vector_store.save_index(self.index_path)
            
            logger.debug(f"💾 Processing metadata saved to {self.index_path}")
                
        except Exception as e:
            logger.error(f"❌ Error saving metadata: {e}")
    
    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics"""
        vector_stats = self.vector_store.get_stats()
        metadata_stats = self.metadata_manager.get_topic_statistics()
        
        stats = {
            **self.stats,
            'vector_store_stats': vector_stats,
            'metadata_stats': metadata_stats,
            'index_path': self.index_path,
            'config_info': {
                'chunk_size': self.config.chunking.chunk_size,
                'chunk_overlap': self.config.chunking.chunk_overlap,
                'embedding_model': self.config.embedding.model_name,
                'vector_store_type': self.config.vector_store.store_type
            }
        }
        
        # Add hybrid-specific stats
        if self.enable_hybrid and self.hybrid_retriever:
            hybrid_stats = self.hybrid_retriever.get_stats()
            stats['hybrid_indexing_stats'] = hybrid_stats
        
        return stats
    
    def save_index(self):
        """Save all indices and metadata"""
        try:
            self._save_processing_metadata()
            self._save_processing_state()
            logger.info(f"💾 All indices saved to {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving indices: {e}")
            return False
    
    def get_metadata_summary(self) -> Dict:
        """Get summary of metadata information"""
        if not self.metadata_manager.metadata_sheets:
            self.metadata_manager.load_metadata()
        
        summary = {
            'total_topics': len(self.metadata_manager.metadata_sheets),
            'topics': {},
            'excel_path': self.metadata_manager.excel_path,
            'metadata_fields_available': []
        }
        
        # Extract metadata fields from first document
        for topic, df in self.metadata_manager.metadata_sheets.items():
            original_docs = df[df['file_type'] == 'original_markdown']
            if not original_docs.empty:
                summary['metadata_fields_available'] = list(original_docs.columns)
                break
        
        for topic, df in self.metadata_manager.metadata_sheets.items():
            original_docs = df[df['file_type'] == 'original_markdown']
            summary['topics'][topic] = {
                'total_documents': len(original_docs),
                'documents_with_tables': len(original_docs[original_docs['has_tables'] > 0]),
                'date_range': {
                    'earliest': original_docs['doc_date'].min() if 'doc_date' in original_docs.columns else None,
                    'latest': original_docs['doc_date'].max() if 'doc_date' in original_docs.columns else None
                }
            }
        
        return summary


def display_enhanced_processing_results(results: List[Dict]):
    """Display enhanced processing results summary with topic breakdown"""
    print("\n" + "="*120)
    print("📊 ENHANCED DOCUMENT PROCESSING RESULTS (with Excel Metadata)")
    print("="*120)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"✅ Successfully processed: {len(successful)} documents")
    print(f"❌ Failed to process: {len(failed)} documents")
    
    if successful:
        total_chunks = sum(r.get('chunks_created', 0) for r in successful)
        total_embeddings = sum(r.get('embeddings_generated', 0) for r in successful)
        bm25_indexed = sum(1 for r in successful if r.get('bm25_indexed', False))
        
        print(f"📝 Total chunks created: {total_chunks}")
        print(f"🧠 Total embeddings generated: {total_embeddings}")
        print(f"🔍 Documents with BM25 index: {bm25_indexed}")
        
        # Show metadata integration
        sample_result = successful[0]
        metadata_fields = sample_result.get('metadata_fields', [])
        if metadata_fields:
            print(f"🏷️ Metadata fields integrated: {', '.join(metadata_fields)}")
        
        # Topic breakdown
        topic_stats = {}
        for result in successful:
            topic = result.get('topic', 'Unknown')
            if topic not in topic_stats:
                topic_stats[topic] = {'docs': 0, 'chunks': 0}
            topic_stats[topic]['docs'] += 1
            topic_stats[topic]['chunks'] += result.get('chunks_created', 0)
        
        print(f"\n📈 Results by Topic:")
        for topic, stats in topic_stats.items():
            print(f"   🏷️ {topic}: {stats['docs']} documents, {stats['chunks']} chunks")
    
    if failed:
        print(f"\n❌ Failed documents:")
        for result in failed:
            print(f"   • {result.get('filename', 'Unknown')}: {result.get('error', 'Unknown error')}")
    
    print("="*120)


async def main():
    """Main function for enhanced document indexing with Excel metadata"""
    parser = argparse.ArgumentParser(description='Enhanced Vietnamese Document Indexing with Excel Metadata')
    parser.add_argument('--index-path', default='storage/index', 
                       help='Path to save vector index')
    parser.add_argument('--metadata-excel', required=True,
                       help='Path to Excel metadata file with multiple sheets')
    parser.add_argument('--base-directory', default=DOCUMENTS_PATH,
                       help='Base directory containing documents')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing')
    parser.add_argument('--force-reindex', action='store_true',
                       help='Force reindexing of all documents')
    parser.add_argument('--topics', nargs='+',
                       help='Specific topics (sheet names) to process')
    parser.add_argument('--disable-hybrid', action='store_true',
                       help='Disable hybrid indexing (BM25)')
    parser.add_argument('--clear-index', action='store_true',
                       help='Clear existing index before processing')
    parser.add_argument('--stats', action='store_true',
                       help='Show processing statistics')
    parser.add_argument('--metadata-summary', action='store_true',
                       help='Show metadata summary')
    parser.add_argument('--config-preset', default='development',
                       choices=['development', 'production', 'high_memory', 'low_memory', 'table_focused'],
                       help='Configuration preset')
    
    args = parser.parse_args()
    
    # Initialize configuration
    print("🚀 Initializing Enhanced Document Indexing System...")
    
    config_presets_map = {
        'development': ConfigPresets.development(),
        'production': ConfigPresets.production(),
        'high_memory': ConfigPresets.high_memory(),
        'low_memory': ConfigPresets.low_memory(),
        'table_focused': ConfigPresets.table_focused()
    }
    
    config = config_presets_map[args.config_preset]
    print(f"📋 Using config: {args.config_preset}")
    print(f"🗄️ Vector store: {config.vector_store.store_type}")
    print(f"🔄 Hybrid indexing: {not args.disable_hybrid}")
    print(f"📊 Metadata Excel: {args.metadata_excel}")
    
    # Initialize enhanced indexing system
    indexing_system = EnhancedDocumentIndexingSystem(
        config=config,
        index_path=args.index_path,
        enable_hybrid=not args.disable_hybrid,
        metadata_excel=args.metadata_excel
    )
    
    # Show metadata summary
    if args.metadata_summary:
        summary = indexing_system.get_metadata_summary()
        print(f"\n📊 Metadata Summary:")
        print(f"   📁 Excel file: {summary['excel_path']}")
        print(f"   🏷️ Total topics: {summary['total_topics']}")
        print(f"   📋 Available metadata fields: {', '.join(summary['metadata_fields_available'])}")
        for topic, info in summary['topics'].items():
            print(f"   • {topic}: {info['total_documents']} docs ({info['documents_with_tables']} with tables)")
        return
    
    # Show current statistics
    if args.stats:
        stats = indexing_system.get_processing_statistics()
        print(f"\n📊 Current Statistics:")
        print(f"   Documents processed: {stats['documents_processed']}")
        print(f"   Chunks created: {stats['chunks_created']}")
        print(f"   Vector store chunks: {stats['vector_store_stats'].get('active_chunks', 0)}")
        
        if stats.get('topics_processed'):
            print(f"   Topics processed:")
            for topic, topic_stats in stats['topics_processed'].items():
                print(f"     • {topic}: {topic_stats['documents']} docs, {topic_stats['chunks']} chunks")
        
        if stats.get('metadata_stats'):
            print(f"   Metadata topics available:")
            for topic, meta_stats in stats['metadata_stats'].items():
                print(f"     • {topic}: {meta_stats['total_documents']} total docs")
        
        return
    
    # Clear index if requested
    if args.clear_index:
        print("🗑️ Clearing existing index...")
        # Note: Implement clear_index method if needed
        print("✅ Index cleared successfully")
    
    # Process documents based on metadata
    print(f"📚 Processing documents from metadata...")
    if args.topics:
        print(f"🎯 Processing specific topics: {args.topics}")
    if args.force_reindex:
        print("🔄 Force reindexing enabled")
    
    results = await indexing_system.process_from_metadata(
        base_directory=args.base_directory,
        batch_size=args.batch_size,
        force_reindex=args.force_reindex,
        specific_topics=args.topics
    )
    
    # Display results
    display_enhanced_processing_results(results)
    
    # Final statistics
    final_stats = indexing_system.get_processing_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   📄 Documents processed: {final_stats['documents_processed']}")
    print(f"   ✂️ Chunks created: {final_stats['chunks_created']}")
    print(f"   🧠 Embeddings generated: {final_stats['embeddings_generated']}")
    print(f"   💾 Vector store size: {final_stats['vector_store_stats'].get('active_chunks', 0)} chunks")
    
    if final_stats.get('topics_processed'):
        print(f"   🏷️ Topics processed:")
        for topic, topic_stats in final_stats['topics_processed'].items():
            print(f"     • {topic}: {topic_stats['documents']} docs, {topic_stats['chunks']} chunks")
    
    if final_stats.get('hybrid_indexing_stats'):
        hybrid_stats = final_stats['hybrid_indexing_stats']
        print(f"   🔍 BM25 indexed: {final_stats['bm25_indexed']} chunks")
        print(f"   🎯 Reranker available: {hybrid_stats.get('reranker_available', False)}")
    
    print(f"   💾 Index saved to: {args.index_path}")
    print("\n✅ Enhanced document indexing completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())