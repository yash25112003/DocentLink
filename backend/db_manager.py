from pymongo import MongoClient
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        # MongoDB connection string
        self.connection_string = "mongodb+srv://yashshah25112003:uQDoppAs6b74ipTD@ta-ra-assistant.djncl4r.mongodb.net/?retryWrites=true&w=majority&appName=TA-RA-ASSISTANT"
        self.client = MongoClient(self.connection_string)
        self.db = self.client['ask_my_prof']  # Main database
        self.collection = self.db.user_profiles  # Single collection for all user data

    def create_user_profile(self, user_data):
        """Create a new user profile in the database."""
        user_data['created_at'] = datetime.utcnow()
        user_data['updated_at'] = datetime.utcnow()
        result = self.collection.insert_one(user_data)
        return result.inserted_id

    def get_user_profile(self, user_id):
        """Retrieve a user profile from the database."""
        return self.collection.find_one({'user_id': user_id})

    def update_user_profile(self, user_id, update_data):
        """Update a user profile in the database."""
        update_data['updated_at'] = datetime.utcnow()
        result = self.collection.update_one(
            {'user_id': user_id},
            {'$set': update_data}
        )
        return result.modified_count > 0

    def delete_user_profile(self, user_id):
        """Delete a user profile from the database."""
        result = self.collection.delete_one({'user_id': user_id})
        return result.deleted_count > 0

    def get_all_profiles(self):
        """Get all user profiles from the database."""
        return list(self.collection.find())

    def get_profiles_by_status(self, status):
        """Get all profiles with a specific status."""
        return list(self.collection.find({'status': status}))

    def create_professor_profile(self, professor_data):
        """Create a new professor profile in the database."""
        collection = self.db.professor_profiles
        professor_data['created_at'] = datetime.utcnow()
        result = collection.insert_one(professor_data)
        return result.inserted_id

    def get_professor_profile(self, professor_id):
        """Retrieve a professor profile from the database."""
        collection = self.db.professor_profiles
        return collection.find_one({'_id': professor_id})

    def create_university(self, university_data):
        """Create a new university in the database."""
        collection = self.db.universities
        university_data['created_at'] = datetime.utcnow()
        result = collection.insert_one(university_data)
        return result.inserted_id

    def get_university(self, university_id):
        """Retrieve a university from the database."""
        collection = self.db.universities
        return collection.find_one({'_id': university_id})

    def create_conversation(self, conversation_data):
        """Create a new conversation in the database."""
        collection = self.db.conversations
        conversation_data['created_at'] = datetime.utcnow()
        result = collection.insert_one(conversation_data)
        return result.inserted_id

    def get_user_conversations(self, user_id):
        """Retrieve all conversations for a user."""
        collection = self.db.conversations
        return list(collection.find({'user_id': user_id}).sort('created_at', -1))

    def close_connection(self):
        """Close the MongoDB connection."""
        self.client.close()

# Example usage in your Streamlit app:
"""
from db_manager import DatabaseManager

# Initialize the database manager
db = DatabaseManager()

# Create a new user profile
user_data = {
    'name': 'John Doe',
    'email': 'john@example.com',
    'university': 'Example University'
}
user_id = db.create_user_profile(user_data)

# Create a professor profile
professor_data = {
    'name': 'Dr. Smith',
    'department': 'Computer Science',
    'university_id': 'some_university_id',
    'research_interests': ['AI', 'Machine Learning']
}
professor_id = db.create_professor_profile(professor_data)

# Close the connection when done
db.close_connection()
""" 