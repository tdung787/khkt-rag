"""
build_db.py

Script to rebuild database from scratch
- Deletes old database files
- Creates new database with all tables
- Initializes all managers to ensure tables are created
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.tools.quiz_storage import QuizStorage
from src.tools.submission_manager import SubmissionManager
from src.tools.session_manager import SessionManager
from src.tools.chat_history_manager import ChatHistoryManager


# Database path
DB_PATH = "database/quiz_storage.db"
DB_DIR = "database"


def delete_old_database():
    """Delete old database files"""
    files_to_delete = [
        DB_PATH,
        f"{DB_PATH}-wal",
        f"{DB_PATH}-shm"
    ]
    
    deleted = []
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted.append(file_path)
            print(f"   🗑️  Deleted: {file_path}")
    
    if deleted:
        print(f"\n✅ Deleted {len(deleted)} old database file(s)")
    else:
        print("\n📭 No old database files found")


def create_database_directory():
    """Ensure database directory exists"""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        print(f"✅ Created directory: {DB_DIR}")
    else:
        print(f"📁 Directory exists: {DB_DIR}")


def initialize_tables():
    """Initialize all tables by creating manager instances"""
    print("\n🔧 Initializing database tables...")
    
    try:
        # Initialize all managers (they will create tables)
        print("   📊 Creating quiz_storage tables...")
        quiz_storage = QuizStorage(db_path=DB_PATH)
        
        print("   📝 Creating submission tables...")
        submission_manager = SubmissionManager(db_path=DB_PATH)
        
        print("   💬 Creating session tables...")
        session_manager = SessionManager(db_path=DB_PATH)
        
        print("   📜 Creating chat_history tables...")
        chat_history_manager = ChatHistoryManager(db_path=DB_PATH)
        
        print("\n✅ All tables created successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing tables: {e}")
        return False


def verify_database():
    """Verify database was created correctly"""
    if not os.path.exists(DB_PATH):
        print("\n❌ Database file not found!")
        return False
    
    import sqlite3
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = [
            'chat_messages',
            'chat_sessions',
            'quizzes',
            'submissions',
            'students'
        ]
        
        print("\n📋 Database tables:")
        for table in tables:
            status = "✅" if table in expected_tables else "⚠️"
            print(f"   {status} {table}")
        
        missing = set(expected_tables) - set(tables)
        if missing:
            print(f"\n⚠️  Missing tables: {missing}")
            return False
        
        print(f"\n✅ All {len(expected_tables)} expected tables found!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error verifying database: {e}")
        return False


def main():
    """Main build process"""
    print("=" * 70)
    print("DATABASE REBUILD SCRIPT")
    print("=" * 70)
    
    # Step 1: Delete old database
    print("\n📍 STEP 1: Deleting old database...")
    delete_old_database()
    
    # Step 2: Create directory
    print("\n📍 STEP 2: Ensuring database directory exists...")
    create_database_directory()
    
    # Step 3: Initialize tables
    print("\n📍 STEP 3: Initializing database tables...")
    success = initialize_tables()
    
    if not success:
        print("\n❌ BUILD FAILED!")
        sys.exit(1)
    
    # Step 4: Verify
    print("\n📍 STEP 4: Verifying database...")
    verified = verify_database()
    
    if not verified:
        print("\n⚠️  BUILD COMPLETED WITH WARNINGS!")
        sys.exit(1)
    
    # Success
    print("\n" + "=" * 70)
    print("🎉 DATABASE REBUILD COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📍 Database location: {os.path.abspath(DB_PATH)}")
    print(f"📊 File size: {os.path.getsize(DB_PATH) / 1024:.2f} KB")
    print("\n💡 You can now start the API server:")
    print("   python run_api.py")
    print()


if __name__ == "__main__":
    # Confirm before deleting
    print("\n⚠️  WARNING: This will DELETE all existing data!")
    confirm = input("Are you sure you want to rebuild the database? (yes/no): ")
    
    if confirm.lower() in ['yes', 'y']:
        main()
    else:
        print("\n❌ Cancelled by user")