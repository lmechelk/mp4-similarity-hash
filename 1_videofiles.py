from pathlib import Path
import pandas as pd

# ==========================================
# EXTRACT VIDEO FILE PATHS
# ==========================================

def extract_video_paths(start_path: str, output_csv: str):
    """
    Recursively finds video files and saves their absolute paths to a CSV.
    input: start_path (str) - directory to search for video files 
              output_csv (str) - path to save the resulting CSV file
    output: CSV file with a single column "path" containing absolute paths of video files
    """
    
    # Convert input path to Path object for easier handling
    base_dir = Path(start_path)
    
    # Only mp4 and mov files are relevant for our use case
    target_exts = {'.mp4', '.mov'}
    
    # Recursively find files, resolve absolute paths, filter by extension
    video_paths = [
        str(p.resolve()) for p in base_dir.rglob('*') 
        if p.is_file() and p.suffix.lower() in target_exts
    ]

    # Convert list to pandas DataFrame and export with semicolon separator
    df = pd.DataFrame(video_paths, columns=["path"])
    df.to_csv(output_csv, sep=';', index=False, encoding='utf-8')
    
    # Print summary of results
    print(f"Found {len(video_paths)} video files. Exported to: {output_csv}")

if __name__ == "__main__":
    # ==========================================
    # TESTDATA
    # ==========================================

    start_dir = r"G:\Meine Ablage\Cybermaster\5_WT26\Masterarbeit\Testdaten\SELFMADE"
    output_csv = r".\1_videofiles\1_testdaten1.csv"   
    extract_video_paths(start_dir, output_csv)

    start_dir = r"G:\Meine Ablage\Cybermaster\5_WT26\Masterarbeit\Testdaten\EVA-7K"
    output_csv = r".\1_videofiles\1_testdaten2.csv"
    extract_video_paths(start_dir, output_csv)

    start_dir = r"G:\Meine Ablage\Cybermaster\5_WT26\Masterarbeit\Testdaten\VISION"
    output_csv = r".\1_videofiles\1_testdaten3.csv"
    extract_video_paths(start_dir, output_csv)
