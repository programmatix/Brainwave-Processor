from google.cloud import storage
import os

# gcloud auth login
# gcloud auth application-default login
# The latter creates C:\Users\graha\AppData\Roaming\gcloud\application_default_credentials.json
# This is the ADC configuration
# $HOME/.config/gcloud/application_default_credentials.json

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
            blob.upload_from_filename(local_path)


def upload_file_to_gcs(log, bucket_name, full_local_filename, prefix):
    """Uploads a directory to a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    local_dir, local_filename = os.path.split(full_local_filename)

    cloud_storage_name = f"{prefix}/{local_filename}"
    blob = bucket.blob(cloud_storage_name)

    log(f"{full_local_filename} uploading to bucket {bucket_name}: {cloud_storage_name}")
    blob.upload_from_filename(full_local_filename)




