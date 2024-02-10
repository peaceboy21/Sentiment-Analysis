from sentimentanalysis.config.configuration import ConfigurationManager
from sentimentanalysis.components.data_ingestion import DataIngestion
from sentimentanalysis.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.extract_zip_file()