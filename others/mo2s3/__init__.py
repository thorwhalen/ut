import os

try:
    DEFAULT_ACCESS_KEY = os.getenv('VEN_S3_ACCESS_KEY')
    DEFAULT_SECRET_KEY = os.getenv('VEN_S3_SECRET')
    DEFAULT_BUCKET = 'mongo-db-bak'
    DEFAULT_FOLDER = ''
except Exception:
    DEFAULT_ACCESS_KEY = ''
    DEFAULT_SECRET_KEY = ''
    DEFAULT_BUCKET = ''
    DEFAULT_FOLDER = ''

default_conf = """[aws]
access_key = {access_key}
secret_key = {secret_key}
s3_bucket = {s3_bucket}
folder = {folder}

[mongodb]
host = 
username = 
password = """.format(
    access_key=DEFAULT_ACCESS_KEY, secret_key=DEFAULT_SECRET_KEY,
    s3_bucket=DEFAULT_BUCKET, folder=DEFAULT_FOLDER)