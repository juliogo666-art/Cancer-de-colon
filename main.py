from src.scripts.eda import eda
import os

def main():
    base_path = os.path.dirname(__file__)
    eda(base_path)
            
if __name__ == "__main__":
    main()
