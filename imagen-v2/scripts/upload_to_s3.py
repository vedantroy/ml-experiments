#! /usr/bin/env python3

# Could just use the aws s3 cli, but
# by writing a script, we need to setup the target environment less
import os
from dotenv import load_dotenv
from pathlib import Path

import boto3
from fastargs import Section, Param
from fastargs.decorators import param
from tqdm import tqdm

from dataset_writer_utils import init_cli_args

Section("s3", "s3 info").params(
    bucket=Param(str, "The bucket name", default=None)
)

Section("files", "files on your disk").params(
    data_dir=Param(str, "where your files are on disk")
)

load_dotenv()

endpoint_url = os.environ.get("S3_ENDPOINT_URL")
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

#s3 = boto3.resource('s3',
#  endpoint_url = endpoint_url,
#  aws_access_key_id = aws_access_key_id,
#  aws_secret_access_key = aws_secret_access_key
#)

s3_client = boto3.client("s3", 
  endpoint_url = endpoint_url,
  aws_access_key_id = aws_access_key_id,
  aws_secret_access_key = aws_secret_access_key
)


@param("s3.bucket", "bucket_name")
@param("files.data_dir")
def run(bucket_name, data_dir):
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"{data_dir} must be a directory"

    # bucket = s3.Bucket(bucket_name)
    files = list(data_dir.glob("*"))
    for file in tqdm(files):
        name = file.name
        s3_client.upload_file(str(file), bucket_name, name)

init_cli_args("Upload a dataset to a S3-compatible API")
run()