# app/snowflake.py
import os
import snowflake.connector

TOKEN_PATH = "/snowflake/session/token"

def get_snowflake_connection():
    with open(TOKEN_PATH) as f:
        token = f.read().strip()

    return snowflake.connector.connect(
        host=os.getenv("SNOWFLAKE_HOST"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        authenticator="oauth",
        token=token,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

