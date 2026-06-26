import os
import sys
import mmap
import struct
import json
import pandas as pd

# ==========================================
# CONSTANTS
# ==========================================

# Valid types, the extended forensic list of mp4 container atoms
CONTAINERS = {

    # Base ISO/IEC 14496-12
    b'moov', b'trak', b'mdia', b'minf', b'stbl', b'udta', b'meta', 
    b'edts', b'dinf', b'stsd', b'mvex', b'moof', b'traf', b'mfra',

    # Extended ISO/IEC 14496-14 and -15
    b'mp4a', b'avc1', b'hev1',

    # Quick Time Legacy
    b'gmhd', b'clip', b'matt', b'wave'
}

# ==========================================
# PARSER
# ==========================================

class Mp4Parser:
    """
    Parses a single MP4 file and builds a deep nested dictionary tree 
    representing its internal box structure.

    input: file path to an MP4 file
    output: a dictionary containing the filename, parsing status, statistics,
    """
    def __init__(self, file_path):
        
        # Extract filename, parent directory, and grandparent directory
        self.file_path = file_path
        name = os.path.basename(file_path)
        parent = os.path.basename(os.path.dirname(file_path))
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        if grandparent and parent:
            self.filename = f"{grandparent}/{parent}/{name}"
        else:
            self.filename = file_path
        
        self.tree = []             # This will hold the nested structure of boxes as a list of dictionaries
        self.signature_string = "" # This will hold the flattened string representation of the structure (nice to have)
        self.statistics = {}       # Statistics about the parsing process (e.g., total boxes, presence of 'uuid', parsing errors)
        
    def parse(self):
        """
        Executes the parsing process and populates the tree and statistics.
        Returns a dictionary representing the processing result.
        """
        try:
            file_size = os.path.getsize(self.file_path)
            
            with open(self.file_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    self.tree = self._parse_boxes(mm, 0, file_size)
                    
            # Generate the string representation from the nested tree (just for nice output, not used for matching)
            self.signature_string = "|".join(self._build_flat_string(self.tree))
            
            # Calculate statistics (e.g., total boxes, presence of 'uuid', parsing errors)
            clean_str = self.signature_string.replace('|', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
            self.statistics = {
                "total_boxes": len(clean_str) // 4,
                "contains_uuid": "uuid" in self.signature_string,
                "parsing_errors": "...." in self.signature_string
            }
            
            # Return the structured result
            return {
                "filename": self.filename,
                "status": "success",
                "statistics": self.statistics,
                "signature": self.signature_string,
                "tree": self.tree
            }
            
        # Handle files that are corrupt, not found, or entirely unreadable
        except Exception as e:

            return {
                "filename": self.filename,
                "status": "error",
                "error_message": str(e),
                "statistics": {"total_boxes": 0, "contains_uuid": False, "parsing_errors": True},
                "signature": "",
                "tree": []
            }

    def _parse_boxes(self, mm, start_offset, end_offset):
        """
        Recursively parses valid MP4 boxes (as in CONTAINERS) and returns a list of dictionaries (nodes).
        The standard header is 8 bytes (size + type), but we also handle extended size (16 bytes) and 
        special cases for certain box types (e.g., 'meta', 'stsd', 'avc1', 'hev1', 'mp4a').

        input:
        - mm: memory-mapped file object for efficient access    
        - start_offset: where to start parsing in the file
        - end_offset: where to stop parsing for this level of recursion
        output: a list of dictionaries representing the boxes at this level, with nested 'children' for containers
        """
        nodes = []              # This will hold the nodes at the current level of recursion
        offset = start_offset   # We start parsing from the given offset and continue until we reach the end_offset for this level
        
        # The main parsing loop: we read box headers, determine their size and type, and then either recurse into them 
        # (if they are containers) or just add them as leaf nodes.
        while offset < end_offset:
            if end_offset - offset < 8:
                break
            
            # Read the box header (size and type) with the standard 8-byte header, and handle extended size if needed
            # First 4 bytes: box size (32-bit unsigned int), next 4 bytes: box type (4-character code)
            box_header = mm[offset:offset+8]
            box_size, box_type = struct.unpack('>I4s', box_header)
            header_size = 8
            
            # Handle special cases for box size (0 means until end of file, 1 means extended size in the next 8 bytes)
            if box_size == 0:
                box_size = end_offset - offset
            
            # Handle 64-bit Large Box (Size = 1) critical for huge mdat atoms
            elif box_size == 1:
                if end_offset - offset < 16:
                    break
                box_size = struct.unpack('>Q', mm[offset+8:offset+16])[0]
                header_size = 16

            # Sanity check: box size must be at least the header size
            if box_size < header_size:
                break
            
            # Sanity check: box must fit within the remaining bytes of this level
            if box_type == b'\x00\x00\x00\x00':
                break

            box_type_str = box_type.decode('latin1', errors='replace')

            # Failsafe: Offset desynchronization
            # Debug only: Add "...." a 4CC as placeholder
            if '\x00' in box_type_str:
                print(f"Warning: Parsing error (null bytes in atom) -> {self.file_path}")
                # nodes.append({"type": "...."})
                # Skip this box and try to resync at the next header (move forward by at least 8 bytes to avoid infinite loop)
                offset += max(box_size if box_size >= 8 else 8, 8)
                continue 

            inner_start = offset + header_size

            # Special handling for certain box types to skip version/flags or other non-structural bytes
            # meta: skip 4 bytes if they are all zeros (version/flags)
            if box_type == (b'meta'):
                peek_bytes = mm[inner_start:inner_start+4]
                if peek_bytes == b'\x00\x00\x00\x00':
                    inner_start += 4
            
            # stsd: skip 8 bytes (version/flags + entry count)
            elif box_type == b'stsd':
                inner_start += 8
            
            # avc1/hev1: skip 78 bytes of reserved fields and codec-specific data
            elif box_type in (b'avc1', b'hev1'):
                inner_start += 78
            
            # mp4a: Handle QuickTime specific Audio Sample Entry versions
            elif box_type == b'mp4a':
                # The version field is located exactly 16 bytes after the box start
                version = struct.unpack('>H', mm[offset+16:offset+18])[0]
                
                if version == 1:
                    inner_start += 44  # QuickTime V1 adds 16 extra bytes
                elif version == 2:
                    inner_start += 64  # QuickTime V2 adds 36 extra bytes
                else:
                    inner_start += 28  # Standard ISO / QuickTime V0

            # Build Node (dictionary) for the current box
            node = {"type": box_type_str}

            # Recursion if container
            if box_type in CONTAINERS:
                children = self._parse_boxes(mm, inner_start, offset + box_size)
                if children:
                    node["children"] = children

            nodes.append(node)
            offset += box_size
        
        return nodes

    def _build_flat_string(self, nodes):
        """
        Recursively flattens the deep dictionary tree back into the familiar 
        topology string format (e.g., "moov(mvhd,trak(tkhd))"), just for display purposes.

        input: a list of nodes (dictionaries) at the current level
        output: a list of strings representing the box types at this level, with nested structures for containers
        """
        parts = []
        for node in nodes:
            if "children" in node:
                children_str = ",".join(self._build_flat_string(node["children"]))
                parts.append(f"{node['type']}({children_str})")
            else:
                parts.append(node["type"])
        return parts


# ==========================================
# NEW PANDAS INTEGRATION 
# ==========================================

def process_dataset_with_pandas(input_csv, output_csv):
    """
    Reads the dataset via pandas, extracts tree/signature via Mp4Parser, 
    and exports it back as a structured CSV.
    """
    print(f"Loading dataset from: {input_csv}")
    
    # Read CSV with auto-detection for the separator (handles ',' or ';')
    try:
        df = pd.read_csv(input_csv, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Pre-allocate lists for new columns
    pretty_string_col = []
    json_col = []
    
    total_files = len(df)
    
    # Iterate through DataFrame rows
    for index, row in df.iterrows():
        path = row.get('path', '')
        
        # Simple progress tracking
        if (index + 1) % 500 == 0 or (index + 1) == total_files:
            print(f"Parsing... {index + 1}/{total_files}")
            
        # Check validity of file path
        if pd.isna(path) or not os.path.isfile(str(path)):
            print(f"Warning: File not found -> {path}")
            pretty_string_col.append("")  # Empty on error
            json_col.append("[]")         # Empty array on error
            continue
            
        # Init original parser and extract data
        parser = Mp4Parser(str(path))
        result = parser.parse()
        
        # Process parser results
        if result["status"] == "success":
            pretty_string_col.append(result["signature"])
            # Dump dictionary tree to minified JSON string
            json_col.append(json.dumps(result["tree"], separators=(',', ':')))
        else:
            print(f"Warning: Parsing failed for {path} -> {result.get('error_message')}")
            pretty_string_col.append("")
            json_col.append("[]")

    # Append new columns to DataFrame
    df['structure_pretty'] = pretty_string_col
    df['structure_json'] = json_col
    
    # Export to new CSV, enforcing semicolon
    df.to_csv(output_csv, sep=';', index=False)
    print(f"Processed {total_files} files. Exported to: {output_csv}")


if __name__ == "__main__":
    # ==========================================
    # TESTDATA
    # ==========================================

    INPUT_CSV = r".\2_datasets\2_testdaten.csv"
    OUTPUT_CSV = r".\3_parse\3_testdaten.csv"
    
    process_dataset_with_pandas(INPUT_CSV, OUTPUT_CSV)