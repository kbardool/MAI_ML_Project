
import pymongo

__author__ = 'KBardool'

class Database(object):
    # URI = os.environ.get("MONGOLAB_URI")
    URI = "mongodb://127.0.0.1:27017"
    DATABASE= None

    @staticmethod
    def initialize():
        client = pymongo.MongoClient(Database.URI)
        # Database.DATABASE = client.get_default_database()
        Database.DATABASE = client['cache']
        print('*** Database initialization successful.',type(client))
#         print('    Client[client]:',client['cache'])
#         print('    Database is ',Database.DATABASE)
        return

    @staticmethod
    def insert(collection, data):
        print('   Database.insert: Collection is',collection,' Data is ', data)
        rc = Database.DATABASE[collection].insert(data)
        print(' Datavase rc is :',rc)

    @staticmethod
    def delete(collection, query):
        print('   Database.remove: Collection is',collection,' Data is ', query)
        Database.DATABASE[collection].remove(query)

    @staticmethod
    def find(collection, query):
        print('   Database.find: Collection is',collection,' Query is ', query)
        p = Database.DATABASE[collection].find(query)
        if p is not None:
            print('   Database.find found: ')
        else:
            print('   Database.find: Data row not found ')
        return p


    @staticmethod
    def find_one(collection, query):
        print('   Database.find_one: Collection is',collection,' Query is ', query)
        p = Database.DATABASE[collection].find_one(query)
        if p is not None:
            print('   Database.find_one: ', p)
        else:
            print('   Database.find_one: Data row not found ')
        return p


    @staticmethod
    def update(collection, query, data):
#         print('   Database.update: Collection is', collection, ' Data is ', data)
        rc = Database.DATABASE[collection].update(query, data, upsert=True)
        print(' Datavase rc is :',rc)