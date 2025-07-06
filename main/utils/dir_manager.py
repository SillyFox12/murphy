import os
import re

class DirectoryCreator:
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        self.output_dir = self.create_next_session_dir()

    # Create the next session(x) folder
    def create_next_session_dir(self):
        self.ensure_directory(self.base_dir)

        # List existing session directories
        existing_sessions = [
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d)) and re.match(r"session\d+", d)
        ]

        # Extract numbers and find the max
        session_numbers = [
            int(re.search(r"session(\d+)", name).group(1))
            for name in existing_sessions
        ] if existing_sessions else []

        next_number = max(session_numbers, default=0) + 1
        next_session_name = f"session{next_number}"
        next_session_path = os.path.join(self.base_dir, next_session_name)

        self.ensure_directory(next_session_path)
        return next_session_path

    # Ensure a directory exists
    def ensure_directory(self, path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            raise
        print(f"Directory ensured: {path}")

    # Return the session directory path
    def get_output_dir(self):
        # Replace backslashes with forward slashes for consistency across platforms
        return self.output_dir.replace(os.sep, '/')     

if __name__ == "__main__":
    dc = DirectoryCreator()
    print("Created:", dc.get_output_dir())
    # Example usage