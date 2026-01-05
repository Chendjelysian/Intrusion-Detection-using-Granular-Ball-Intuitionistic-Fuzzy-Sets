#!/usr/bin/env python3
"""
Download preprocessed GBIFS datasets from Google Drive and extract to ./datatest/
"""

import os
import zipfile
from pathlib import Path
import gdown

# === 配置区 ===
FILE_ID = "1VAxs4vjlNsSmenlO-vxmFt4_5ZbJrC8y"  # ← 你的文件ID
OUTPUT_ZIP = "data.zip"
DATA_DIR = "data"  # 目标目录（审稿人看到的路径）


# ==============

def main():
    # 创建目标目录
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)

    zip_path = data_path / OUTPUT_ZIP

    print("📥 Downloading dataset from Google Drive...")
    try:
        gdown.download(
            id=FILE_ID,
            output=str(zip_path),
            quiet=False,
            fuzzy=True  # 自动处理 confirm 页面
        )
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Make sure the file is shared publicly on Google Drive.")
        return

    if not zip_path.exists():
        print("❌ ZIP file not found after download. Check the file ID.")
        return

    print("📦 Extracting files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print(f"✅ Extraction complete! Data is in './{DATA_DIR}/'")
    except zipfile.BadZipFile:
        print("❌ Downloaded file is corrupted or not a ZIP archive.")
        return

    # 可选：删除 ZIP 文件
    zip_path.unlink()
    print("🗑️  Temporary ZIP file removed.")


if __name__ == "__main__":
    main()