from cnnClassifier.components.data_splitter import create_data_split
from cnnClassifier import logger

STAGE_NAME = "Data Splitting"

class DataSplittingPipeline:
    def __init__(self):
        pass

    def main(self):
        """Executa a divisão dos dados em train/val/test"""
        stats = create_data_split()
        
        if stats:
            logger.info("Pipeline de divisão de dados concluída com sucesso")
            return stats
        else:
            raise Exception("Falha na divisão dos dados")

if __name__ == "__main__":
    try:
        logger.info(f"***" * 10)
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataSplittingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
