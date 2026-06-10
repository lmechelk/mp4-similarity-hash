import pandas as pd
from pathlib import Path
import combine
import os

# ==========================================
# CONSTANTS
# ==========================================

# Dictionarie for mapping id names to brand and model
id_MAP = {
    # db: vision, eva
    'D01': ('Samsung', 'GalaxyS3Mini'), 
    'D02': ('Apple', 'iPhone4s'),
    'D03': ('Huawei', 'P9'), 
    'D04': ('LG', 'D290'), 
    'D05': ('Apple', 'iPhone5c'),
    'D06': ('Apple', 'iPhone6'), 
    'D07': ('Lenovo', 'P70A'), 
    'D08': ('Samsung', 'GalaxyTab3'),
    'D09': ('Apple', 'iPhone4'), 
    'D10': ('Apple', 'iPhone4s'), 
    'D11': ('Samsung', 'GalaxyS3'),
    'D12': ('Sony', 'XperiaZ1Compact'), 
    'D13': ('Apple', 'iPad2'), 
    'D14': ('Apple', 'iPhone5c'),
    'D15': ('Apple', 'iPhone6'), 
    'D16': ('Huawei', 'P9lite'), 
    'D17': ('Microsoft', 'Lumia640LTE'),
    'D18': ('Apple', 'iPhone5c'), 
    'D19': ('Apple', 'iPhone6Plus'), 
    'D20': ('Apple', 'iPadmini'),
    'D21': ('Wiko', 'Ridge4G'), 
    'D22': ('Samsung', 'GalaxyTrendPlus'), 
    'D23': ('Asus', 'Zenfone2Laser'),
    'D24': ('Xiaomi', 'RedmiNote3'), 
    'D25': ('OnePlus', 'A3000'), 
    'D26': ('Samsung', 'GalaxyS3Mini'),
    'D27': ('Samsung', 'GalaxyS5'), 
    'D28': ('Huawei', 'P89'), 
    'D29': ('Apple', 'iPhone5'),
    'D30': ('Huawei', 'Honor5'), 
    'D31': ('Samsung', 'GalaxyS4Mini'), 
    'D32': ('OnePlus', 'A3003'),
    'D33': ('Huawei', 'AscendG6'), 
    'D34': ('Apple', 'iPhone5'), 
    'D35': ('Samsung', 'GalaxyTabA'),
    
    # db: selfmade_ai
    'AI01': ('Grok', 'Imagine_0.9'),
    'AI02': ('Kling', 'KlingAIVideo_2.6'),
    'AI03': ('LumaAI', 'LumaDream'),
    'AI04': ('Pika', 'Pika_2.5'),
    'AI05': ('RunwayAI', 'Runway_4.5'),
    'AI06': ('Bing', 'Sora_2.0'),
    'AI07': ('Gemini', 'Veo_3.1'),
    'AI08': ('Perplexity', 'Veo_3.1')
}

# Dictionarie for mapping processing types
PROCESSING_MAP = {
    # db: vision, eva
    'flat': ('FLAT', 'ORIGINAL'),
    'flatWA': ('FLAT', 'WHATSAPP'),
    'flatYT': ('FLAT', 'YOUTUBE'),
    'indoor': ('INDOOR', 'ORIGINAL'), 
    'indoorWA': ('INDOOR', 'WHATSAPP'),
    'indoorYT': ('INDOOR', 'YOUTUBE'),
    'outdoor': ('OUTDOOR', 'ORIGINAL'),
    'outdoorWA': ('OUTDOOR', 'WHATSAPP'),
    'outdoorYT': ('OUTDOOR', 'YOUTUBE'), 
    
    # db: selfmade_ai
    'synth': ('SYNTHETIC', 'ORIGINAL'), # Videocreation only with a text prompt
    'df': ('DEEPFAKE', 'ORIGINAL')      # Created with an uploaded picture as a reference
}

# ==========================================
# LOADING & SAVING
# ==========================================

def _load_dataset(input_csv: str) -> pd.DataFrame:
    """Lädt die CSV und führt grundlegende Validierungen durch."""
    if not Path(input_csv).is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    df = pd.read_csv(input_csv, sep=';')
    if 'path' not in df.columns:
        raise ValueError("Critical column 'path' missing.")
    return df

def _save_dataset(df: pd.DataFrame, output_csv: str):
    """Sortiert die Spalten und speichert den DataFrame konsistent als CSV."""
    expected_cols = ['path', 'db_name', 'id', 'brand', 'model', 'content_source', 'content_type', 'media_type', 'processing', 'tampering']
    print(f"{len(df)} entries processed. Exported to: {output_csv}")
    df[expected_cols].to_csv(output_csv, index=False, sep=';')


# ==========================================
# DB: VISON
# ==========================================

def process_vision(input_csv: str, output_csv: str):

    # Load dataset
    df = _load_dataset(input_csv)

    # Extract metadata from paths
    paths = df['path'].str.replace('\\', '/', regex=False).apply(Path)
    splits = paths.apply(lambda p: p.stem.split('_'))
    df['id'] = splits.str[0]
    context = splits.str[2]
    
    # Lookup based on id and context
    df['brand'] = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[0])
    df['model'] = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[1])
    df['content_type'] = context.map(lambda c: PROCESSING_MAP.get(c, (pd.NA, pd.NA))[0])
    df['processing'] = context.map(lambda c: PROCESSING_MAP.get(c, (pd.NA, pd.NA))[1])
    df['media_type'] = paths.apply(lambda p: p.suffix[1:].upper())

    # Static infos
    df['tampering'] = 'NONE'
    df['db_name'] = 'VISION'
    df['content_source'] = 'DIGITAL_CAMERA'
    
    # Save dataset
    _save_dataset(df, output_csv)

# ==========================================
# DB: EVA
# ==========================================

def process_eva(input_csv: str, output_csv: str):

    # Load dataset
    df = _load_dataset(input_csv)

    # Extract metadata from paths
    paths = df['path'].str.replace('\\', '/', regex=False).apply(Path)
    splits = paths.apply(lambda p: p.stem.split('_'))
    df['id'] = splits.str[0]
    
    # Lookup based on id and context
    df['brand'] = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[0])
    df['model'] = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[1])
    df['content_type'] = splits.str[2].str.upper()
    df['processing'] = paths.apply(lambda p: p.parent.parent.name.upper())
    df['tampering'] = paths.apply(lambda p: p.parent.name.upper().replace('SUBSET_', '').replace('ORIGINAL', 'NONE'))
    df['media_type'] = paths.apply(lambda p: p.suffix[1:].upper())
    
    # Static infos
    df['db_name'] = 'EVA-7K'
    df['content_source'] = 'DIGITAL_CAMERA'
    
    # Save dataset
    _save_dataset(df, output_csv)

# ==========================================
# DB: SELFMADE APPLE
# ==========================================

def process_selfmade_apple(input_csv: str, output_csv: str):

    # Load dataset
    df = _load_dataset(input_csv)

    # Extract metadata from paths
    paths = df['path'].str.replace('\\', '/', regex=False).apply(Path)
    df['id'] = paths.apply(lambda p: p.parent.name.upper())
    df['model'] = paths.apply(lambda p: p.parent.parent.name)
    df['media_type'] = paths.apply(lambda p: p.suffix[1:].upper())
    
    # Static infos
    df['brand'] = 'APPLE'
    df['content_type'] = 'NONE'
    df['processing'] = 'ORIGINAL'
    df['tampering'] = 'NONE'
    df['db_name'] = 'SELFMADE'
    df['content_source'] = 'DIGITAL_CAMERA'
    
    # Save dataset
    _save_dataset(df, output_csv)

# ==========================================
# DB: SELFMADE AI
# ==========================================

def process_selfmade_ai(input_csv: str, output_csv: str):

    # Load dataset
    df = _load_dataset(input_csv)

    # Extract metadata from paths
    paths = df['path'].str.replace('\\', '/', regex=False).apply(Path)
    splits = paths.apply(lambda p: p.stem.split('_'))
    df['id'] = splits.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else pd.NA)
    context = splits.apply(lambda x: x[3] if isinstance(x, list) and len(x) > 3 else pd.NA)
    df['media_type'] = paths.apply(lambda p: p.suffix[1:].upper())
    
    # Lookup
    df['brand'] = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[0])
    df['model'] = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[1])
    df['content_type'] = context.map(lambda c: PROCESSING_MAP.get(c, (pd.NA, pd.NA))[0])
    df['processing'] = context.map(lambda c: PROCESSING_MAP.get(c, (pd.NA, pd.NA))[1])

    # Content source based on content type
    df.loc[df['content_type'] == 'SYNTHETIC', 'content_source'] = 'AI_GENERATION'
    df.loc[df['content_type'] == 'DEEPFAKE', 'content_source'] = 'MIXED'
       
    # Static
    df['tampering'] = 'NONE'
    df['db_name'] = 'SELFMADE'

    # Save dataset
    _save_dataset(df, output_csv)


if __name__ == "__main__":
    # ==========================================
    # TESTDATA
    # ==========================================

    process_vision(
        r".\1_videofiles\1_testdaten3.csv", 
        r".\2_datasets\2_testdaten3.csv")

    process_eva(
        r".\1_videofiles\1_testdaten2.csv", 
        r".\2_datasets\2_testdaten2.csv")   
    
    process_selfmade_ai(
        r".\1_videofiles\1_testdaten1.csv", 
        r".\2_datasets\2_testdaten1.csv")
    
    FILE_1 = r".\2_datasets\2_testdaten1.csv"
    FILE_2 = r".\2_datasets\2_testdaten2.csv"
    OUTPUT = r".\2_datasets\2_testdaten.csv"
    combine.combine_csv_files(FILE_1, FILE_2, OUTPUT)

    FILE_1 = r".\2_datasets\2_testdaten3.csv"
    FILE_2 = r".\2_datasets\2_testdaten.csv"
    OUTPUT = r".\2_datasets\2_testdaten.csv"
    combine.combine_csv_files(FILE_1, FILE_2, OUTPUT)

    
    #process_selfmade_apple("1_selfmade_apple.csv", "2_selfmade_apple.csv")
    