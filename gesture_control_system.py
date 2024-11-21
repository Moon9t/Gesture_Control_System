import os
import sys
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

# Project directory structure
PROJECT_STRUCTURE = {
    "data": [],
    "models": [],
    "scripts": [],
    "logs": [],
    "assets": ["instructions.txt", "sample_gesture_images/"],
}

def print_logo():
    """
    Display a logo with ownership name 'MOON9T' at the start of the program.
    """
    logo = """
     __  __           ____    ____  _____  _______  _______ 
    |  \/  |   /\    / __ \  / __ \|  __ \|__   __|/ ____|
    | \  / |  /  \  | |  | || |  | | |__) |  | |  | (___  
    | |\/| | / /\ \ | |  | || |  | |  _  /   | |   \___ \ 
    | |  | |/ ____ \| |__| || |__| | | \ \   | |   ____) |
    |_|  |_/_/    \_\\____/  \____/|_|  \_\  |_|  |_____/ 
                                                             
    """
    print(Fore.YELLOW + logo)
    print(Fore.GREEN + "Welcome to Gesture Control System by MOON9T\n")

def initialize_project():
    """
    Create the project directory structure if it doesn't already exist.
    """
    print(Fore.CYAN + "Initializing project structure...")
    
    for folder, sub_items in PROJECT_STRUCTURE.items():
        # Create the main folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"{Fore.GREEN}Created folder: {folder}")
        
        # Create subdirectories or files inside the folder
        for item in sub_items:
            item_path = os.path.join(folder, item)
            if item.endswith("/"):  # It's a subdirectory
                if not os.path.exists(item_path):
                    os.makedirs(item_path)
                    print(f"{Fore.GREEN}Created subdirectory: {item_path}")
            else:  # It's a file
                if not os.path.exists(item_path):
                    with open(item_path, "w") as f:
                        if item == "instructions.txt":
                            f.write("Welcome to the Gesture Control System!\n")
                            f.write("1. Collect gesture data.\n")
                            f.write("2. Train the gesture recognition model.\n")
                            f.write("3. Use real-time recognition to perform actions.\n")
                        print(f"{Fore.GREEN}Created file: {item_path}")

def main_menu():
    """
    Display the main menu and handle user input.
    """
    print(Fore.MAGENTA + "\n===== Gesture Control System =====")
    print(Fore.CYAN + "1. Collect Gesture Dataset")
    print(Fore.CYAN + "2. Train Gesture Recognition Model")
    print(Fore.CYAN + "3. Run Real-Time Gesture Recognition")
    print(Fore.CYAN + "4. Exit")
    choice = input(Fore.YELLOW + "\nEnter your choice (1-4): ")
    return choice

def main():
    # Print the logo and ownership information
    print_logo()

    # Initialize the project structure
    initialize_project()

    # Import scripts dynamically to avoid circular imports
    try:
        from scripts.dataset_acquisition import collect_data
        from scripts.train_model import train_gesture_model
        from scripts.real_time_recognition import recognize_gestures
    except ImportError as e:
        print(f"{Fore.RED}Error: {e}")
        sys.exit(1)

    while True:
        choice = main_menu()

        if choice == "1":
            print(Fore.BLUE + "\nStarting Dataset Acquisition...")
            gesture_name = input(Fore.YELLOW + "Enter gesture name to collect (e.g., swipe_left): ").strip()
            num_samples = int(input(Fore.YELLOW + "Enter the number of samples to collect: "))
            collect_data(gesture_name, num_samples)
            print(Fore.GREEN + f"Dataset for gesture '{gesture_name}' collected successfully!")

        elif choice == "2":
            print(Fore.BLUE + "\nTraining the Gesture Recognition Model...")
            model_path = train_gesture_model()
            print(Fore.GREEN + f"Model trained successfully! Saved to: {model_path}")

        elif choice == "3":
            print(Fore.BLUE + "\nStarting Real-Time Gesture Recognition...")
            recognize_gestures()

        elif choice == "4":
            print(Fore.RED + "Exiting the program. Goodbye!")
            sys.exit()

        else:
            print(Fore.RED + "Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
