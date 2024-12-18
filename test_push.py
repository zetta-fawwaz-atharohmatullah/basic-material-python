from astrapy import DataAPIClient
from dotenv import load_dotenv

load_dotenv(override=True)


# client = DataAPIClient("TOKEN")
# database0 = client.get_database("DB_API_ENDPOINT")
# collection0 = database0.create_collection("movies", dimension=5)
# collection0.insert_one({
#     "title": "The Title",
#     "$vector": [0.1, 0.3, -0.7, 0.9, -0.1],
# })
# admin = client.get_admin()
# database_iterator = admin.list_databases()



from astrapy import DataAPIClient

client = DataAPIClient("AstraCS:EUvtNkSSkwCLDyMrYUHrbTAb:d649ebe3107705415da9517bd580f1777c8a25fd8d1f3d5ed3c7632d4fe9e0eb")
db = client.get_database_by_api_endpoint(
  "https://15a08ff8-98c9-4838-909b-d022aacbd38d-us-east-2.apps.astra.datastax.com",
    keyspace="default_keyspace",
)
      
print(f"Connected to Astra DB: {db.list_collection_names()}")