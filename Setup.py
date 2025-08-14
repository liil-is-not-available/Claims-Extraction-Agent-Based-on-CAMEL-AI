import numpy as np
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.logger import get_logger
def setup():
    """Using 'model_backend = setup()' to set up"""
    api = np.loadtxt('APIs.csv', delimiter=',', dtype=str, skiprows=1)
    openai_api_key = str(api[0][1])
    chunkr_api_key = str(api[1][1])
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_api_key.strip()
    os.environ["CHUNKR_API_KEY"] = chunkr_api_key.strip()

    logger = get_logger(__name__)

    model_backend = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict={
            "stream": False,
        },
    )
    return model_backend