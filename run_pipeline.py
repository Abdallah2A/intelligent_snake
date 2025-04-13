import os
import logging
from zenml import __version__ as zenml_version
from pipelines.pipeline import snake_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    """
    Initialize ZenML and run the snake_pipeline.
    """
    try:
        # Check if ZenML repository is initialized
        if not os.path.exists(".zen"):
            logger.info("Initializing ZenML repository...")
            os.system("zenml init")
        else:
            logger.info("ZenML repository already initialized.")

        logger.info("Running snake_pipeline with ZenML version %s", zenml_version)

        # Run the pipeline
        snake_pipeline(True)

        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise e


if __name__ == "__main__":
    run()
