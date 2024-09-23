import boto3
import logging

logger = logging.getLogger()


def get_secret(secret_name):

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager")

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response["SecretString"]
        return secret
    except Exception as e:
        logger.error("Error getting secret", exc_info=True)
        raise e


def get_MongoDB_Uri():
    mongo_uri = get_secret("workshop/atlas_secret")
    return mongo_uri
