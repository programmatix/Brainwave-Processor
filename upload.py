import hashlib

from google.cloud import storage
import os
from google.cloud.exceptions import NotFound


# gcloud auth login
# gcloud auth application-default login
# The latter creates
# Windows: %USERPROFILE%\AppData\Roaming\gcloud\application_default_credentials.json
# Linux: $HOME/.config/gcloud/application_default_credentials.json
# This is the ADC configuration
# Copy that to the target server:
# scp %USERPROFILE%\AppData\Roaming\gcloud\application_default_credentials.json username@hostname:.config/gcloud/

import hashlib
import base64
from google.cloud import storage
import os
from google.cloud.exceptions import NotFound

def calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calculate_directory_md5(directory_path):
    """Calculate the MD5 checksum of a directory."""
    md5_list = []
    for root, _, files in os.walk(directory_path):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            file_md5 = calculate_md5(file_path)
            md5_list.append(file_md5)
    combined_md5 = hashlib.md5("".join(md5_list).encode('utf-8')).hexdigest()
    return combined_md5

def base64_to_hex(base64_str):
    """Convert a base64-encoded string to a hexadecimal string."""
    return base64.b64decode(base64_str).hex()

def upload_dir_to_gcs_skipping_existing(log, bucket_name, source_dir, destination_blob_prefix):
    """Uploads a directory to a GCS bucket, skipping identical files."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Calculate directory checksum
    directory_md5 = calculate_directory_md5(source_dir)
    checksum_blob_path = os.path.join(destination_blob_prefix, "directory_checksum.md5").replace("\\", "/")
    checksum_blob = bucket.blob(checksum_blob_path)

    try:
        # Get the existing checksum
        checksum_blob.reload()
        remote_md5 = checksum_blob.download_as_text().strip()

        # Compare checksums
        if directory_md5 == remote_md5:
            log(f"Skipping upload of {source_dir}, identical directory already exists in GCS.")
            return
    except NotFound:
        # Checksum blob does not exist, proceed with upload
        log(f"Directory checksum for {source_dir} does not exist in GCS, proceeding with upload.")
        pass

    # Proceed with file-by-file upload
    for root, _, files in os.walk(source_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_dir)
            blob_path = os.path.join(destination_blob_prefix, relative_path).replace("\\", "/")

            blob = bucket.blob(blob_path)

            # Calculate local file checksum
            local_md5 = calculate_md5(local_path)

            has_changed = True
            does_not_exist = False

            try:
                # Get the blob's checksum
                blob.reload()
                remote_md5 = base64_to_hex(blob.md5_hash)

                # Compare checksums
                if local_md5 == remote_md5:
                    log(f"Skipping {local_path}, identical file already exists in GCS.")
                    has_changed = False
                    continue
            except NotFound:
                # Blob does not exist, proceed with upload
                does_not_exist = True
                pass

            log(f"{local_path} uploading to {blob_path}, does_not_exist={does_not_exist} has_changed={has_changed}")
            blob.upload_from_filename(local_path, timeout=600)

    # Upload the new directory checksum
    checksum_blob.upload_from_string(directory_md5)
    log(f"Uploaded directory checksum for {source_dir} to {checksum_blob_path}")

def upload_dir_to_gcs(log, bucket_name, source_dir, destination_blob_prefix):
    """Uploads a directory to a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(source_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_dir)
            blob_path = os.path.join(destination_blob_prefix, relative_path).replace("\\", "/")

            blob = bucket.blob(blob_path)

            log(f"{local_path} uploading to {blob_path}")
            blob.upload_from_filename(local_path, timeout=600)


def upload_file_to_gcs(log, bucket_name, full_local_filename, prefix):
    """Uploads a file to a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    local_dir, local_filename = os.path.split(full_local_filename)

    cloud_storage_name = f"{prefix}/{local_filename}"
    blob = bucket.blob(cloud_storage_name)

    log(f"{full_local_filename} uploading to bucket {bucket_name}: {cloud_storage_name}")
    blob.upload_from_filename(full_local_filename, timeout=600)
