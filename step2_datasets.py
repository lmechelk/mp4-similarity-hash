import pandas as pd
from pathlib import Path
import combine
import os

# ==========================================
# CONSTANTS
# ==========================================

# Dictionary for mapping id names to brand and model
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

    # db: ai
    'AI01': ('GoogleFlow', 'OmniFlash'),
    'AI02': ('GoogleGemini', 'Veo3.1'),
    'AI03': ('Grok', 'Imagine0.9'),
    'AI04': ('Grok', 'Imagine1.5'),
    'AI05': ('KlingAI', 'KlingAIVideo2.6'),
    'AI06': ('KlingAI', 'KlingAIVideo3.0'),
    'AI07': ('MicrosoftBing', 'Sora2'),
    'AI08': ('PikaLabs', 'Pika1.5'),
    'AI09': ('PikaLabs', 'Pika2.0'),
    'AI10': ('PikaLabs', 'Pika2.2'),
    'AI11': ('PikaLabs', 'Pika2.5'),
}

# Dictionary for mapping processing types
PROCESSING_MAP = {
    # db: vision, eva
    'flat':      ('FLAT',    'ORIGINAL'),
    'flatWA':    ('FLAT',    'WHATSAPP'),
    'flatYT':    ('FLAT',    'YOUTUBE'),
    'indoor':    ('INDOOR',  'ORIGINAL'),
    'indoorWA':  ('INDOOR',  'WHATSAPP'),
    'indoorYT':  ('INDOOR',  'YOUTUBE'),
    'outdoor':   ('OUTDOOR', 'ORIGINAL'),
    'outdoorWA': ('OUTDOOR', 'WHATSAPP'),
    'outdoorYT': ('OUTDOOR', 'YOUTUBE'),

    # db: ai — new filename scheme: AI01_01_synth.mp4 / AI01_03_df.mp4
    'synth': ('SYNTHETIC', 'ORIGINAL'),  # text-to-video
    'df':    ('DEEPFAKE',  'ORIGINAL'),  # image-to-video
}

_SM_VALUES = {'WHATSAPP', 'YOUTUBE', 'FACEBOOK', 'TIKTOK', 'WEIBO'}

# ==========================================
# LOADING & SAVING
# ==========================================

def _load_dataset(input_csv: str) -> pd.DataFrame:
    """Load CSV and run basic validation."""
    if not Path(input_csv).is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv, sep=';')
    if 'path' not in df.columns:
        raise ValueError("Critical column 'path' missing.")
    return df

def _save_dataset(df: pd.DataFrame, output_csv: str):
    """Sort columns and save DataFrame consistently as CSV."""
    expected_cols = ['path', 'db_name', 'id', 'brand', 'model', 'content_source',
                     'content_type', 'media_type', 'processing', 'processing_sm', 'tampering']
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    print(f"{len(df)} entries processed. Exported to: {output_csv}")
    df[expected_cols].to_csv(output_csv, index=False, sep=';')

# ==========================================
# DB: VISION
# ==========================================

def process_vision(input_csv: str, output_csv: str):

    # Load dataset
    df = _load_dataset(input_csv)

    # Extract metadata from paths
    paths  = df['path'].str.replace('\\\\', '/', regex=False).apply(Path)
    splits = paths.apply(lambda p: p.stem.split('_'))
    df['id']  = splits.str[0]
    context   = splits.str[2]

    # Lookup based on id and context
    df['brand']        = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[0])
    df['model']        = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[1])
    df['content_type'] = context.map(lambda c: PROCESSING_MAP.get(c, (pd.NA, pd.NA))[0])
    df['processing']   = context.map(lambda c: PROCESSING_MAP.get(c, (pd.NA, pd.NA))[1])
    df['media_type']   = paths.apply(lambda p: p.suffix[1:].upper())
    df['processing_sm'] = df['processing'].str.upper().isin(_SM_VALUES)

    # Static infos
    df['tampering']     = 'NATIVE'
    df['db_name']       = 'VISION'
    df['content_source'] = 'DIGITAL_CAMERA'

    _save_dataset(df, output_csv)

# ==========================================
# DB: EVA
# ==========================================

def process_eva(input_csv: str, output_csv: str):

    # Load dataset
    df = _load_dataset(input_csv)

    # Extract metadata from paths
    paths  = df['path'].str.replace('\\\\', '/', regex=False).apply(Path)
    splits = paths.apply(lambda p: p.stem.split('_'))
    df['id'] = splits.str[0]

    # Lookup based on id
    df['brand']        = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[0])
    df['model']        = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[1])
    df['content_type'] = splits.str[2].str.upper()
    df['processing']   = paths.apply(lambda p: p.parent.parent.name.upper())
    df['tampering']    = paths.apply(lambda p: p.parent.name.upper().replace('SUBSET_', '').replace('ORIGINAL', 'NATIVE'))
    df['media_type']   = paths.apply(lambda p: p.suffix[1:].upper())
    df['processing_sm'] = df['processing'].str.upper().isin(_SM_VALUES)

    # Static infos
    df['db_name']       = 'EVA-7K'
    df['content_source'] = 'DIGITAL_CAMERA'

    _save_dataset(df, output_csv)

# ==========================================
# DB: SELFMADE APPLE
# ==========================================

def process_selfmade_apple(input_csv: str, output_csv: str):

    # Load dataset
    df = _load_dataset(input_csv)

    # Extract metadata from paths
    paths = df['path'].str.replace('\\\\', '/', regex=False).apply(Path)
    df['id']         = paths.apply(lambda p: p.parent.name.upper())
    df['model']      = paths.apply(lambda p: p.parent.parent.name)
    df['media_type'] = paths.apply(lambda p: p.suffix[1:].upper())

    # Static infos
    df['brand']         = 'APPLE'
    df['content_type']  = 'NONE'
    df['processing']    = 'ORIGINAL'
    df['tampering']     = 'NATIVE'
    df['db_name']       = 'SELFMADE'
    df['content_source'] = 'DIGITAL_CAMERA'
    df['processing_sm'] = 'FALSE'

    _save_dataset(df, output_csv)

# ==========================================
# DB: AI 
# ==========================================

def process_selfmade_ai(input_csv: str, output_csv: str):

    # Load dataset
    df = _load_dataset(input_csv)

    # Extract metadata from paths
    paths  = df['path'].str.replace('\\\\', '/', regex=False).apply(Path)
    splits = paths.apply(lambda p: p.stem.split('_'))

    # parts[0] = AI-ID, parts[2] = content type keyword
    df['id']        = splits.apply(lambda x: x[0] if len(x) > 0 else pd.NA)
    context          = splits.apply(lambda x: x[2] if len(x) > 2 else pd.NA)
    df['media_type'] = paths.apply(lambda p: p.suffix[1:].upper())

    # Lookup
    df['brand']        = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[0])
    df['model']        = df['id'].map(lambda d: id_MAP.get(d, (pd.NA, pd.NA))[1])
    df['content_type'] = context.map(lambda c: PROCESSING_MAP.get(c, (pd.NA, pd.NA))[0])
    df['processing']   = context.map(lambda c: PROCESSING_MAP.get(c, (pd.NA, pd.NA))[1])

    # Content source based on content type
    df['content_source'] = pd.NA
    df.loc[df['content_type'] == 'SYNTHETIC', 'content_source'] = 'AI_GENERATION'
    df.loc[df['content_type'] == 'DEEPFAKE',  'content_source'] = 'MIXED'

    # Static
    df['tampering'] = 'NATIVE'
    df['db_name']   = 'AI'
    df['processing_sm'] = 'FALSE'

    _save_dataset(df, output_csv)

if __name__ == "__main__":

    process_selfmade_ai(
            r".\1_videofiles\1_ai.csv", 
            r".\2_datasets\2_ai.csv")

