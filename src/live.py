import boto3
import os
from pathlib import Path
from botocore.exceptions import NoCredentialsError
from pyparsing import C
from ray.tune import Callback
import time
import threading
from dvclive import Live
import fsspec


def list_objects_in_s3_dir(bucket_name, s3_dir):
    """
    List all objects in a specific directory of an S3 bucket using fsspec.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - s3_dir (str): The directory within the S3 bucket.
    """
    # Initialize the filesystem
    fs = fsspec.filesystem('s3')
    
    # Ensure the s3_dir ends with a slash for consistency
    if not s3_dir.endswith('/'):
        s3_dir += '/'

    s3_path = f's3://{bucket_name}/{s3_dir}'

    print(f"Listing objects in '{s3_path}':")

    # Use fsspec to list all objects
    files = fs.ls(s3_path)

    # Remove the directory path to get only the object names
    obj_keys = [file.split('/')[-1] for file in files if not file.endswith('/')]

    return obj_keys


def download_from_s3(bucket_name, s3_path, local_path):
    """
    Downloads a file or directory from an S3 directory.

    :param bucket_name: The name of the S3 bucket.
    :param s3_path: The directory or file within the S3 bucket.
    :param local_path: The local path where to save the downloaded file or directory.
    """
    fs = fsspec.filesystem('s3')
    s3_path_base = f"s3://{bucket_name}/{s3_path.strip('/')}/"  # Ensure proper formatting
    print("s3_path_base ", s3_path_base)
    
    if not fs.exists(s3_path_base):
        raise ValueError(f"The S3 path {s3_path_base} does not exist.")

    if fs.isfile(s3_path_base):
        # Download a single file
        fs.get(s3_path_base, local_path)
        print(f"Downloaded {s3_path_base} to {local_path}")
        
    else:
        # Download a directory
        files = fs.find(s3_path_base)
        for file in files:

            # Compute relative path correctly
            relative_path = os.path.relpath(file.split(bucket_name)[1].strip('/'), s3_path)            
            local_file_path = os.path.join(local_path, relative_path)

            # Ensure the directory structure is created in the local filesystem
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            fs.get(file, local_file_path)
            print(f"Downloaded {file} to {local_file_path}")



def upload_to_s3(local_path, bucket_name, s3_dir):
    """
    Uploads a file or directory to an S3 directory.

    :param local_path: Path to the local file or directory to upload.
    :param bucket_name: The name of the S3 bucket.
    :param s3_dir: The directory within the S3 bucket.
    """
    fs = fsspec.filesystem('s3')
    s3_path_base = f"s3://{bucket_name}/{s3_dir.strip('/')}/"  # Ensure proper formatting
    print("s3_path_base ", s3_path_base)
    
    if os.path.isdir(local_path):
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_path).replace("\\", "/")
                s3_path = os.path.join(s3_path_base, relative_path)
                fs.put(file_path, s3_path)
                print(f"Uploaded {file_path} to {s3_path}")
                
    elif os.path.isfile(local_path):
        s3_path = os.path.join(s3_path_base, os.path.basename(local_path))
        fs.put(local_path, s3_path)
        print(f"Uploaded {local_path} to {s3_path}")
    else:
        raise ValueError(f"The path {local_path} does not exist.")


class DVCLiveRayLogger(Live):
    def __init__(self, bucket_name, s3_directory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.s3_directory = s3_directory

    def next_step(self, *args, **kwargs):
        super().next_step(*args, **kwargs)

        print("\nDVCLiveLogger: Push DVCLive metrics to S3")
        upload_to_s3(self.dir, self.bucket_name, self.s3_directory,)
