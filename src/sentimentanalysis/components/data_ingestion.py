import os
import zipfile
import logging
from sentimentanalysis.logging import logger
from sentimentanalysis.utils.common import get_size
from pathlib import Path
from sentimentanalysis.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)