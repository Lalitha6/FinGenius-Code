import os
import subprocess

def main():
    # Set environment variables
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    
    # Run the Streamlit app
    subprocess.run(['streamlit', 'run', 'app.py'])

if __name__ == "__main__":
    main()
