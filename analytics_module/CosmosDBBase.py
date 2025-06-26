import logging
import os # Added for environment variable access
from typing import Optional, List

# Azure SDK Imports
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos import PartitionKey, exceptions as cosmos_exceptions

# --- Logging Setup ---
# Get a logging specific to this module.
# The actual configuration (level, handlers, formatters) should be done
# by the application that uses this module.

# --- End Logging Setup ---

# --- Example: Application-Level Logging Configuration ---
# This block would typically be in your application's main entry point
# (e.g., main.py, app.py) or a dedicated logging configuration module.
# It's included here for illustrative purposes.

# if __name__ == "__main__": # Or in your app's setup
#     # --- Application Logging Configuration ---
#     # Create a formatter
#     log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')

#     # Get the root logging or a specific application logging
#     # app_logging = logging.getlogging() # For root logging
#     # app_logging = logging.getlogging("my_application") # For a specific app logging

#     # Set the logging level from an environment variable, defaulting to INFO
#     log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
#     log_level = getattr(logging, log_level_name, logging.INFO)
#     # For the module logging, if you want to ensure it also respects this:
#     # logging.getlogging(__name__).setLevel(log_level) # Though typically the root/app logging's level is sufficient if this logging propagates

#     # Configure the root logging (or your application's main logging)
#     # This ensures that log messages from this module (CosmosDBBase)
#     # and other parts of your application are handled.
#     root_logging = logging.getlogging()
#     root_logging.setLevel(log_level)

#     # Console Handler (for local development or simple cases)
#     console_handler = logging.StreamHandler()
#     # console_handler.setLevel(log_level) # Handler level can be more specific if needed
#     console_handler.setFormatter(log_formatter)
#     if not root_logging.hasHandlers(): # Add handler only if no handlers are configured
#         root_logging.addHandler(console_handler)

#     # Optional: Azure Monitor Integration (for production in Azure)
#     # from opencensus.ext.azure.log_exporter import AzureLogHandler
#     # AZURE_MONITOR_CONN_STR = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
#     # if AZURE_MONITOR_CONN_STR:
#     #     azure_handler = AzureLogHandler(connection_string=AZURE_MONITOR_CONN_STR)
#     #     azure_handler.setFormatter(log_formatter) # Can use the same or a different formatter
#     #     root_logging.addHandler(azure_handler)
#     #     logging.info("Azure Monitor logging configured.")
#     # else:
#     #     logging.info("Azure Monitor connection string not found. Skipping Azure Monitor logging.")

#     logging.info(f"Application logging configured with level: {log_level_name}")
#     # --- End Application Logging Configuration ---

# --- Base DAL Class ---
class CosmosDBBase:
    def __init__(self, client: AsyncCosmosClient, db_name: str, container_name: str, automatic_id_generation: bool):
        self.client = client
        self.db_name = db_name
        self.container_name = container_name
        self.database_client = None
        self.container_client = None
        self.automatic_id_generation = automatic_id_generation
        # logging is already defined at the module level
        # self.logging = logging.getlogging(__name__) # No longer needed here if using module logging directly

    async def initialize_container(self, partition_key_path: str):
        """Initialize database and container if they don't exist."""
        try:
            self.database_client = self.client.get_database_client(self.db_name)
            await self.database_client.read()
            logging.info(f"Database '{self.db_name}' found.")
        except cosmos_exceptions.CosmosHttpResponseError as e:
            if e.status_code == 404:
                logging.warning(f"Database '{self.db_name}' not found. Creating...")
                try:
                    self.database_client = await self.client.create_database(id=self.db_name)
                    logging.info(f"Database '{self.db_name}' created.")
                except cosmos_exceptions.CosmosHttpResponseError as e_create_db:
                    logging.error(f"Failed to create database '{self.db_name}': {e_create_db}")
                    raise
            else:
                logging.error(f"Error accessing database '{self.db_name}': {e}")
                raise

        try:
            self.container_client = self.database_client.get_container_client(self.container_name)
            await self.container_client.read()
            logging.info(f"Container '{self.container_name}' found.")
        except cosmos_exceptions.CosmosHttpResponseError as e:
            if e.status_code == 404:
                logging.warning(f"Container '{self.container_name}' not found. Creating with partition key '{partition_key_path}'...")
                try:
                    self.container_client = await self.database_client.create_container(
                        id=self.container_name,
                        partition_key=PartitionKey(path=partition_key_path)
                        # Add indexing policy, throughput settings etc. here if needed
                    )
                    logging.info(f"Container '{self.container_name}' created.")
                except cosmos_exceptions.CosmosHttpResponseError as e_create_cont:
                    logging.error(f"Failed to create container '{self.container_name}': {e_create_cont}")
                    raise
            else:
                logging.error(f"Error accessing container '{self.container_name}': {e}")
                raise

    # --- Standard CRUD (Keep as is, they are fine) ---
    async def create_item(self, item: dict):
        if not self.container_client:
            logging.error("Container client not initialized before create_item call.")
            raise RuntimeError("Container client not initialized. Call initialize_container first.")
        return await self.container_client.create_item(body=item, enable_automatic_id_generation=self.automatic_id_generation)

    async def read_item(self, item_id: str, partition_key: str):
        if not self.container_client:
            logging.error("Container client not initialized before read_item call.")
            raise RuntimeError("Container client not initialized. Call initialize_container first.")
        try:
            return await self.container_client.read_item(item=item_id, partition_key=partition_key)
        except cosmos_exceptions.CosmosHttpResponseError as e:
            if e.status_code == 404:
                logging.warning(f"Item '{item_id}' with partition key '{partition_key}' not found.")
                return None
            else:
                logging.error(f"Error reading item '{item_id}': {e}")
                raise
        return None # Should be unreachable if logic is correct, but defensive

    async def query_items(self, query: str, parameters: Optional[List[dict]] = None, partition_key: Optional[str] = None):
        if not self.container_client:
            logging.error("Container client not initialized before query_items call.")
            raise RuntimeError("Container client not initialized. Call initialize_container first.")
        items = []
        try:
            query_iterable = self.container_client.query_items(
                query=query,
                parameters=parameters,
                partition_key=partition_key
            )
            async for item in query_iterable:
                items.append(item)
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logging.error(f"Error querying items with query '{query}': {e}")
            raise
        return items

    async def upsert_item(self, item: dict):
        if not self.container_client:
            logging.error("Container client not initialized before upsert_item call.")
            raise RuntimeError("Container client not initialized. Call initialize_container first.")
        return await self.container_client.upsert_item(body=item)

    async def delete_item(self, item_id: str, partition_key: str):
        if not self.container_client:
            logging.error("Container client not initialized before delete_item call.")
            raise RuntimeError("Container client not initialized. Call initialize_container first.")
        try:
            return await self.container_client.delete_item(item=item_id, partition_key=partition_key)
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logging.error(f"Error deleting item '{item_id}': {e}")
            raise   