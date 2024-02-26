import toml
import os
import boto3

from pathlib import Path
from loguru import logger as log

__USE_LOCAL__ = os.environ.get("LOCAL_MODELS", "False") == "True"
__BUCKET_NAME__ = "nitmre-models"
__REMOTE_MODEL_NAME__ = "llm/mixtral-8x7b-instruct-v0.1.Q6_K.gguf"

def downloadDirectoryFroms3(bucket_name, object_name):
    #connect to the S3 bucket 
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)

    matches = list(bucket.objects.filter(Prefix = object_name))

    model_obj = None

    if len(matches) == 0:
        log.critical(f"No objects found in {bucket_name} S3 bucket matching prefix {object_name}. Aborting...")
        exit()
    elif len(matches) == 1:
        log.info(f"{object_name} found in {bucket_name} S3 bucket...")
        model_obj = matches[0]
    elif len(matches) > 1:
        log.info(f"Multiple objects returned matching prefix: {object_name}, only using first... {matches[0].key}")
        model_obj = matches[0]

    model_path = model_obj.key.replace('llm', 'models')

    if os.path.isfile(model_path):
        log.info(f"{model_obj.key} found locally, skipping download...")
    else:
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        bucket.download_file(model_obj.key, model_path)

if __name__ == "__main__":   
    # Download from S3
    if not __USE_LOCAL__:
        log.info(f"Downloading {__REMOTE_MODEL_NAME__} from {__BUCKET_NAME__} S3 Bucket...")
        downloadDirectoryFroms3(__BUCKET_NAME__, __REMOTE_MODEL_NAME__)
        log.info(f'{__REMOTE_MODEL_NAME__} downloaded from {__BUCKET_NAME__} S3 Bucket...')
    else:
        log.info(f'Skipping S3 download, using local model directory...')

    log.info("Starting NITMRE LLM server...")

    conf_file = Path("./config.toml").absolute()
    if conf_file.exists():
        conf = toml.load(str(conf_file))

    else:
        log.critical("Configuration file not found!")
        exit()

    llm_server_conf = conf["llm-server"]

    cmd_str = "./bin/server"
    for opt_k, opt_v in llm_server_conf.items():
        os_key = "LLM_SERVER_" + opt_k.upper().replace("-", "_")
        if os_key in os.environ:
            opt_v = os.environ[os_key]

        if isinstance(opt_v, bool):
            if opt_v:
                cmd_str += f" --{opt_k}"
        else:
            cmd_str += f" --{opt_k} {opt_v} "
    cmd_str = cmd_str.strip()

    log.info(f"Running command: {cmd_str}")
    os.system(cmd_str)
