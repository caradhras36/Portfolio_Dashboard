#!/usr/bin/env python3
"""
Test Supabase connection
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

def test_supabase_connection():
    """Test if we can connect to Supabase"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get credentials
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("❌ Missing Supabase credentials in .env file")
            print("Please create a .env file with:")
            print("SUPABASE_URL=https://your-project-id.supabase.co")
            print("SUPABASE_KEY=your_anon_key_here")
            return False
        
        print(f"🔗 Testing connection to: {url}")
        
        # Create Supabase client
        supabase: Client = create_client(url, key)
        
        # Test connection by trying to read from a table
        # This will fail if tables don't exist, but connection will work
        try:
            result = supabase.table('portfolio_positions').select('*').limit(1).execute()
            print("✅ Supabase connection successful!")
            print("✅ Database tables are accessible")
            return True
        except Exception as e:
            if "relation" in str(e).lower() and "does not exist" in str(e).lower():
                print("✅ Supabase connection successful!")
                print("⚠️  Tables don't exist yet - you need to run database_schema.sql")
                return True
            else:
                print(f"❌ Database error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPossible issues:")
        print("1. Check your SUPABASE_URL and SUPABASE_KEY in .env file")
        print("2. Make sure your Supabase project is active")
        print("3. Check your internet connection")
        return False

if __name__ == "__main__":
    test_supabase_connection()
