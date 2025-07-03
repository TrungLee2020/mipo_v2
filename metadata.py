# metadata.py - Multi-Sheet Excel Metadata Support

from typing import List, Dict
import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetadataManager:
    """Manages document metadata from multi-sheet Excel files"""
    
    def __init__(self, excel_path: str = None, topic_dir_map: Dict[str, str] = None):
        self.excel_path = excel_path
        self.metadata_sheets = {}  # Dict of {topic: DataFrame}
        self.processed_files = {}  # Track processed files with checksums
        self.file_checksums = {}  # Store file checksums for change detection
        self.topic_map_dir = topic_dir_map or {} # Map topics to directories
        
    def load_metadata(self) -> Dict[str, pd.DataFrame]:
        """Load metadata from all sheets in Excel file"""
        if not self.excel_path or not os.path.exists(self.excel_path):
            logger.warning(f"⚠️ Metadata file not found: {self.excel_path}")
            return {}
        
        try:
            # Read all sheets from Excel file
            excel_file = pd.ExcelFile(self.excel_path)
            self.metadata_sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                self.metadata_sheets[sheet_name] = df
                logger.info(f"📊 Loaded sheet '{sheet_name}' with {len(df)} records")
            
            total_docs = sum(len(df) for df in self.metadata_sheets.values())
            logger.info(f"📚 Total loaded: {len(self.metadata_sheets)} topics, {total_docs} documents")
            return self.metadata_sheets
            
        except Exception as e:
            logger.error(f"❌ Error loading metadata: {e}")
            return {}
    
    def get_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"❌ Error calculating checksum for {file_path}: {e}")
            return ""
    
    def has_file_changed(self, file_path: str) -> bool:
        """Check if file has changed since last processing"""
        current_checksum = self.get_file_checksum(file_path)
        if not current_checksum:
            return False
        
        stored_checksum = self.file_checksums.get(file_path)
        return current_checksum != stored_checksum
    
    def update_file_checksum(self, file_path: str):
        """Update stored checksum for a file"""
        checksum = self.get_file_checksum(file_path)
        if checksum:
            self.file_checksums[file_path] = checksum
    
    def get_documents_to_process(self, base_directory: str = None, force_reindex: bool = False) -> List[Dict]:
        """Get list of documents that need processing based on metadata with sanitized data"""
        if not self.metadata_sheets:
            self.load_metadata()
        
        if not self.metadata_sheets:
            return []
        
        documents_to_process = []
        missing_files = []
        
        # Process each topic (sheet)
        for topic, df in self.metadata_sheets.items():
            logger.info(f"🔍 Processing topic: {topic}")
            
            # Filter for original_markdown files only
            original_docs = df[df['file_type'] == 'original_markdown'].copy()
            
            for _, row in original_docs.iterrows():
                filename = row.get('filename', '')
                doc_id = row.get('doc_id', '')
                doc_title = row.get('doc_title', '')
                doc_date_raw = row.get('doc_date', '')
                if hasattr(doc_date_raw, 'strftime') and not pd.isna(doc_date_raw):
                    doc_date = doc_date_raw.strftime('%Y-%m-%d')
                else:
                    doc_date = str(doc_date_raw) if doc_date_raw else 'Unknown'
                has_tables = row.get('has_tables', 0)
                
                # IMPORTANT: Sanitize doc_date here
                doc_date_str = self._sanitize_doc_date(doc_date)
                
                file_path = None
                # File path resolution logic (unchanged)
                if os.path.isabs(filename) and os.path.exists(filename):
                    file_path = filename
                elif base_directory:
                    candidate = os.path.join(base_directory, filename)
                    if os.path.exists(candidate):
                        file_path = candidate
                elif topic in self.topic_map_dir:
                    candidate = os.path.join(self.topic_map_dir[topic], filename)
                    if os.path.exists(candidate):
                        file_path = candidate
                if not file_path and base_directory:
                    matches = list(glob.glob(os.path.join(base_directory, '**', filename), recursive=True))
                    if matches:
                        file_path = matches[0]
                
                # Check if file exists
                if not file_path or not os.path.exists(file_path):
                    logger.warning(f"⚠️ File not found: {filename}")
                    missing_files.append({
                        'file_path': filename,
                        'filename': filename,
                        'doc_id': str(doc_id),  # Ensure string
                        'doc_title': str(doc_title),  # Ensure string
                        'doc_date': doc_date_str,  # Use sanitized date
                        'topic': topic,
                    })
                    continue
                
                # Check if file needs processing
                needs_processing = force_reindex or self.has_file_changed(file_path)
                
                if needs_processing:
                    # Create sanitized document info
                    document_info = {
                        'file_path': file_path,
                        'filename': filename,
                        'doc_id': str(doc_id),  # Ensure string
                        'doc_title': str(doc_title),  # Ensure string
                        'doc_date': doc_date,  # Already converted to string
                        'topic': topic,
                        'has_tables': bool(has_tables),
                        'needs_processing': True
                    }
                    documents_to_process.append(document_info)
                    logger.debug(f"  📄 Queued: {filename}")
                else:
                    logger.debug(f"  ⏭️ Skipped (unchanged): {filename}")
        
        logger.info(f"📋 Found {len(documents_to_process)} documents to process")
        if missing_files:
            logger.warning(f"⚠️ {len(missing_files)} files missing compared to metadata")
        return documents_to_process

    def _sanitize_doc_date(self, doc_date_value) -> str:
        """Convert doc_date to string format to avoid serialization issues"""
        if not doc_date_value or pd.isna(doc_date_value):
            return 'Unknown'
        
        # If already a string, return as is
        if isinstance(doc_date_value, str):
            return doc_date_value.strip()
        
        # Handle pandas Timestamp
        if isinstance(doc_date_value, pd.Timestamp):
            return doc_date_value.strftime('%Y-%m-%d')
        
        # Handle datetime objects
        try:
            import datetime
            if isinstance(doc_date_value, (datetime.datetime, datetime.date)):
                return doc_date_value.strftime('%Y-%m-%d')
        except Exception:
            pass
        
        # Handle numpy datetime
        try:
            if isinstance(doc_date_value, np.datetime64):
                return str(doc_date_value)[:10]  # Get YYYY-MM-DD part
        except Exception:
            pass
        
        # Fallback: convert to string
        try:
            return str(doc_date_value).strip()
        except Exception:
            return 'Unknown'
    
    def get_topic_statistics(self) -> Dict:
        """Get statistics by topic"""
        if not self.metadata_sheets:
            return {}
        
        stats = {}
        for topic, df in self.metadata_sheets.items():
            original_docs = df[df['file_type'] == 'original_markdown']
            # Xử lý cho format doc_date
            if 'doc_date' in original_docs.columns:
                doc_dates = pd.to_datetime(original_docs['doc_date'], errors='coerce')
                earliest = doc_dates.min()
                latest = doc_dates.max()
                # Nếu NaT thì trả về None
                earliest = earliest if not pd.isna(earliest) else None
                latest = latest if not pd.isna(latest) else None
            else:
                earliest = latest = None

            stats[topic] = {
                'total_documents': len(original_docs),
                'documents_with_tables': len(original_docs[original_docs['has_tables'] > 0]),
                'earliest_date': earliest,
                'latest_date': latest            }
        
        return stats
    
    def save_processing_state(self, state_file: str):
        """Save processing state including file checksums"""
        try:
            state = {
                'file_checksums': self.file_checksums,
                'processed_files': self.processed_files,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 Processing state saved to {state_file}")
        except Exception as e:
            logger.error(f"❌ Error saving processing state: {e}")
    
    def load_processing_state(self, state_file: str):
        """Load processing state including file checksums"""
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                self.file_checksums = state.get('file_checksums', {})
                self.processed_files = state.get('processed_files', {})
                
                logger.info(f"📊 Loaded processing state: {len(self.file_checksums)} file checksums")
        except Exception as e:
            logger.error(f"❌ Error loading processing state: {e}")
