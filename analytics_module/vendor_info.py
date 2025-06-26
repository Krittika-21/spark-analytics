"""
ABOUT: VendorInfoDataManager for fetching vendor data from Azure Cosmos DB 
ABOUT: Extends CosmosDBBase to work with 'Vendor Raw Details' container for vendor info
ABOUT: Uses 'Vendor Raw Details' and 'vendor_master_data' containers
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from CosmosDBBase import CosmosDBBase

from dotenv import load_dotenv
load_dotenv()

class VendorInfoDataManager(CosmosDBBase):
    """Manages vendor info data retrieval from Azure Cosmos DB"""
    
    def __init__(self):
        # Azure Cosmos DB configuration from environment
        self.endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
        self.key = os.getenv("AZURE_COSMOS_KEY")
        self.database_name = "Vendors"
        self.container_name = "Vendor Raw Details"
        
        # Default partition key settings
        self.partition_key_path = "/vendor_code"
        self.partition_key_property = "vendor_code"
        self.id_property = "id"
        
        if not self.endpoint or not self.key:
            raise ValueError("AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY environment variables are required")
        
        # Initialize Cosmos client
        self.cosmos_client = AsyncCosmosClient(self.endpoint, self.key)
        
        # Initialize base class
        super().__init__(
            client=self.cosmos_client,
            db_name=self.database_name,
            container_name=self.container_name,
            automatic_id_generation=True
        )
        
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the database and container connection"""
        if not self.is_initialized:
            await self.connect_to_existing_container()
            self.is_initialized = True
            logging.info("VendorInfoDataManager initialized successfully")
    
    async def connect_to_existing_container(self):
        """Connect to existing container without creating it"""
        try:
            # Get database client
            self.database_client = self.cosmos_client.get_database_client(self.database_name)
            await self.database_client.read()
            logging.info(f"✅ Connected to database: {self.database_name}")
            
            # Get container client
            self.container_client = self.database_client.get_container_client(self.container_name)
            properties = await self.container_client.read()
            logging.info(f"✅ Connected to container: {self.container_name}")
            
            # Extract partition key information
            if 'partitionKey' in properties:
                partition_key_info = properties['partitionKey']
                if 'paths' in partition_key_info and len(partition_key_info['paths']) > 0:
                    self.partition_key_path = partition_key_info['paths'][0]
                    self.partition_key_property = self.partition_key_path.lstrip('/')
                    logging.info(f"📋 Partition key: {self.partition_key_path}")
            
        except Exception as e:
            logging.error(f"❌ Failed to connect to container {self.container_name}: {e}")
            raise
    
    async def get_all_vendors(self) -> List[Dict]:
        """Fetch all vendor records from the container"""
        if not self.is_initialized:
            await self.initialize()
        
        query = "SELECT * FROM c"
        
        try:
            results = await self.query_items(query)
            logging.info(f"Retrieved {len(results)} vendor records")
            return results
        except Exception as e:
            logging.error(f"Failed to fetch all vendors: {e}")
            return []
    
    async def get_vendor_master_data(self) -> pd.DataFrame:
        """Fetch data from vendor_master_data container"""
        try:
            print("🔍 Attempting to connect to vendor_master_data container...")
            
            # Create a separate container client for vendor_master_data
            master_container_client = self.database_client.get_container_client("vendor_master_data")
            
            # Query to get VendorCode and TotalNetOrderValue only
            query = "SELECT c.VendorCode, c.TotalNetOrderValue FROM c"
            print(f"🔍 Executing query: {query}")
            
            master_data = []
            async for item in master_container_client.query_items(query=query):
                master_data.append(item)
                # Show first few items for debugging
                if len(master_data) <= 3:
                    print(f"  Sample master data item {len(master_data)}: {item}")
                    
            if master_data:
                print(f"✅ Retrieved {len(master_data)} records from vendor_master_data")
            else:
                print("⚠️ No data found in vendor_master_data container")
                return pd.DataFrame()
            
            # Convert to DataFrame
            master_df = pd.DataFrame(master_data)
            print(f"📋 Master data columns: {list(master_df.columns)}")
            
            return master_df
            
        except Exception as e:
            print(f"❌ Failed to fetch vendor master data: {e}")
            logging.error(f"Failed to fetch vendor master data: {e}")
            return pd.DataFrame()
    
    async def export_to_dataframe(self, filter_registered_only: bool = True) -> pd.DataFrame:
        """Export vendor data joined with master data"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get data from Vendor Raw Details container
            vendors = await self.get_all_vendors()
            
            if not vendors:
                logging.warning("No vendor data found in raw details")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_raw = pd.DataFrame(vendors)
            
            # Map fields from raw details
            field_mapping = {
                'VendorCode': 'Vendor Code',
                'Name1': 'Vendor Name',
                'Country': 'Vendor Country',
                'StreetorHouseNumber': 'Address'
            }
            
            # Apply field mapping
            for old_field, new_field in field_mapping.items():
                if old_field in df_raw.columns:
                    df_raw = df_raw.rename(columns={old_field: new_field})
            
            # Select only the specific columns from raw details
            raw_columns = ['Vendor Code', 'Vendor Name', 'Address', 'Vendor Country']
            available_raw_columns = [col for col in raw_columns if col in df_raw.columns]
            
            if not available_raw_columns:
                logging.warning("None of the desired columns found in raw details")
                return pd.DataFrame()
            
            df_raw_filtered = df_raw[available_raw_columns].copy()
            
            # Get data from vendor_master_data container
            print("🔗 Attempting to fetch master data for joining...")
            df_master = await self.get_vendor_master_data()
            
            if df_master.empty:
                print("⚠️ No master data found, returning raw data only (4 columns)")
                return df_raw_filtered
            
            print(f"✅ Master data loaded: {len(df_master)} records")
            
            # Perform left join on Vendor Code
            print("🔗 Performing left join...")
            df_joined = pd.merge(
                df_raw_filtered,
                df_master,
                left_on='Vendor Code',
                right_on='VendorCode',
                how='left'
            )
            
            print(f"📊 Join result: {len(df_joined)} rows, {len(df_joined.columns)} columns")
            
            # Drop the duplicate VendorCode column from master data
            if 'VendorCode' in df_joined.columns:
                df_joined = df_joined.drop('VendorCode', axis=1)
                print("🗑️ Dropped duplicate VendorCode column")
            
            # Rename master data columns for clarity
            if 'TotalNetOrderValue' in df_joined.columns:
                df_joined = df_joined.rename(columns={'TotalNetOrderValue': 'Total Net Order Value'})
                print("✅ Renamed TotalNetOrderValue → Total Net Order Value")
            
            # Convert Total Net Order Value to numeric
            if 'Total Net Order Value' in df_joined.columns:
                df_joined['Total Net Order Value'] = pd.to_numeric(df_joined['Total Net Order Value'], errors='coerce')
                print("✅ Converted Total Net Order Value to numeric")
            
            # Remove duplicates (keep first occurrence)
            original_count = len(df_joined)
            df_joined = df_joined.drop_duplicates()
            deduplicated_count = len(df_joined)
            duplicates_removed = original_count - deduplicated_count
            
            if duplicates_removed > 0:
                print(f"🗑️ Removed {duplicates_removed} duplicate rows ({original_count} → {deduplicated_count})")
            else:
                print("✅ No duplicate rows found")
            
            print(f"📋 Final columns: {list(df_joined.columns)}")
            logging.info(f"Successfully joined data: {len(df_joined)} vendors with {len(df_joined.columns)} columns")
            return df_joined
            
        except Exception as e:
            logging.error(f"Failed to export joined DataFrame: {e}")
            return pd.DataFrame()
    
    async def get_vendor_summary_stats(self) -> Dict:
        """Get summary statistics about vendors"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get total count
            total_query = "SELECT VALUE COUNT(1) FROM c"
            total_results = await self.query_items(total_query)
            total_count = total_results[0] if total_results else 0
            
            # Get distinct countries
            countries_query = "SELECT DISTINCT c.Country FROM c WHERE IS_DEFINED(c.Country)"
            country_results = await self.query_items(countries_query)
            distinct_countries = [item["Country"] for item in country_results if item.get("Country")]
            
            return {
                "total_vendors": total_count,
                "distinct_countries": sorted(distinct_countries),
                "country_count": len(distinct_countries),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to get vendor summary stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the Cosmos client"""
        if self.cosmos_client:
            await self.cosmos_client.close()
            logging.info("Cosmos DB client closed")

# Main execution for testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("🔍 Connecting to Vendor Raw Details container...")
        
        try:
            # Create the manager
            manager = VendorInfoDataManager()
            print("✅ VendorInfoDataManager created")
            
            # Initialize connection
            await manager.initialize()
            print("✅ Connected to Cosmos DB successfully!")
            
            # Get summary statistics
            print("\n📈 Getting vendor summary statistics...")
            stats = await manager.get_vendor_summary_stats()
            
            if "error" not in stats:
                print(f"📊 Total Vendors: {stats.get('total_vendors', 0)}")
                
                country_list = stats.get('distinct_countries', [])
                if country_list:
                    print(f"🌍 Countries ({len(country_list)}): {', '.join(country_list[:10])}")
                    if len(country_list) > 10:
                        print(f"    ... and {len(country_list) - 10} more countries")
            else:
                print(f"❌ Stats failed: {stats['error']}")
            
            # Export joined data and save as CSV
            print("\n📊 Exporting joined data and saving as CSV...")
            df = await manager.export_to_dataframe(filter_registered_only=False)
            
            if not df.empty:
                print(f"✅ Joined DataFrame created: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Save to CSV file with timestamp
                csv_filename = f"vendor_joined_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_filename, index=False)
                print(f"📁 Joined data saved to: {csv_filename}")
                
                # Show basic summary
                print(f"\n📊 DATA SUMMARY:")
                print(f"• Total vendors: {len(df)}")
                print(f"• Total columns: {len(df.columns)}")
                print(f"• File size: {os.path.getsize(csv_filename) / 1024:.1f} KB")
                
                # Show column names in a clean list
                print(f"\n📋 Available columns ({len(df.columns)}):")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i:2d}. {col}")
                
                # Show data quality info for all columns
                print(f"\n📊 DATA QUALITY:")
                for col in df.columns:
                    non_null_count = df[col].notna().sum()
                    completeness = (non_null_count / len(df)) * 100
                    print(f"• {col}: {non_null_count}/{len(df)} complete ({completeness:.1f}%)")
                
                # Show Total Net Order Value stats if available
                if 'Total Net Order Value' in df.columns:
                    print(f"\n📊 TOTAL NET ORDER VALUE STATS:")
                    non_null_values = df['Total Net Order Value'].notna().sum()
                    if non_null_values > 0:
                        avg_value = df['Total Net Order Value'].mean()
                        max_value = df['Total Net Order Value'].max()
                        min_value = df['Total Net Order Value'].min()
                        print(f"• Records with values: {non_null_values}/{len(df)}")
                        print(f"• Average: ${avg_value:,.2f}")
                        print(f"• Range: ${min_value:,.2f} - ${max_value:,.2f}")
                    else:
                        print(f"• No non-null values found")
                
                # Show countries if available
                if 'Vendor Country' in df.columns:
                    unique_countries = df['Vendor Country'].nunique()
                    print(f"\n🌍 COUNTRIES: {unique_countries} unique countries")
                
                print(f"\n✅ Open '{csv_filename}' in Excel to view the joined vendor data!")
                
            else:
                print("📭 No data found or exported")
            
            # Close connection
            await manager.close()
            print("\n✅ Connection closed successfully!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the main function
    asyncio.run(main())