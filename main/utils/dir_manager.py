#Import dependencies
import os

# Create a directory if it does not exist
class DirectoryCreator:
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = output_dir
        self.ensure_directory(output_dir)

    # Ensures that the specified directory exists, creating it if necessary
    def ensure_directory(self, path=None):
        if not path:
            output_dir = "./data"
            path = output_dir
        else:
            output_dir = path
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            raise
        print(f"Directory ensured: {path}")
    # Returns the output directory
    #Ensures that output directory is universal for all classes
    def get_output_dir(self):
        return self.output_dir