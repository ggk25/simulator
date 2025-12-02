import json
import os
import subprocess
import shutil
import time
import sys

# Configuration
BATCH_SIZES = [1, 5, 25, 50, 100, 150, 200, 250]
CONFIG_FILE = os.path.join('model_inference', 'model_config.json')
INFERENCE_SCRIPT = os.path.join('model_inference', 'inference.py')
OUTPUT_FILE = os.path.join('model_inference', 'latency_energy.xlsx')

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found at {CONFIG_FILE}")
        sys.exit(1)
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def run_simulation():
    print(f"Running simulation...")
    # Ensure previous output is gone to avoid confusion
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
        except PermissionError:
            print(f"Warning: Could not remove existing {OUTPUT_FILE}. It might be open.")
    
    # Run the inference script
    # We use sys.executable to ensure we use the same python interpreter
    result = subprocess.run([sys.executable, INFERENCE_SCRIPT], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running simulation:")
        print(result.stderr)
        return False
    
    # Print a summary of stdout (optional, or just print it all)
    # print(result.stdout) 
    return True

def main():
    print("Starting batch experiments...")
    original_config = load_config()
    
    # Handle both dict and list config formats
    if isinstance(original_config, list):
        config_data = original_config
        config_to_modify = config_data[0]
        is_list = True
    else:
        config_data = original_config
        config_to_modify = config_data
        is_list = False

    # Keep a backup of the original config in memory to restore later
    import copy
    backup_config = copy.deepcopy(config_data)

    try:
        for batch_size in BATCH_SIZES:
            print(f"\n--- Processing Batch Size: {batch_size} ---")
            
            # Update config
            config_to_modify['batch_size'] = batch_size
            
            save_config(config_data)
            
            # Run simulation
            if run_simulation():
                # Construct new filename
                model_name = config_to_modify.get('name', 'Unknown')
                datatype = config_to_modify.get('datatype', 'fp8')
                prefill_len = config_to_modify.get('prefill_length', config_to_modify.get('prefill_lenth', 0))
                decode_len = config_to_modify.get('decode_length', config_to_modify.get('decode_lenth', 0))
                
                # Naming rule: model-name+batch-size+datatype+prefill_length+decode_length
                new_filename = f"{model_name}+{batch_size}+{datatype}+{prefill_len}+{decode_len}.xlsx"
                new_filepath = os.path.join('model_inference', new_filename)
                
                # Rename output file
                if os.path.exists(OUTPUT_FILE):
                    # Remove existing target file if it exists
                    if os.path.exists(new_filepath):
                        try:
                            os.remove(new_filepath)
                        except PermissionError:
                            print(f"Error: Target file {new_filepath} is open. Cannot overwrite.")
                            continue

                    try:
                        shutil.move(OUTPUT_FILE, new_filepath)
                        print(f"Success! Saved result to: {new_filename}")
                    except Exception as e:
                        print(f"Error moving file: {e}")
                else:
                    # Check for timestamped files if the main one isn't found
                    # inference.py logic: latency_energy_{ts}.xlsx
                    # We look for the most recently created file in the directory that matches the pattern
                    print(f"Warning: {OUTPUT_FILE} not found. Checking for timestamped files...")
                    dir_path = os.path.dirname(OUTPUT_FILE)
                    files = [f for f in os.listdir(dir_path) if f.startswith('latency_energy_') and f.endswith('.xlsx')]
                    if files:
                        # Sort by modification time
                        files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=True)
                        latest_file = os.path.join(dir_path, files[0])
                        print(f"Found timestamped file: {files[0]}")
                        
                        if os.path.exists(new_filepath):
                            try:
                                os.remove(new_filepath)
                            except PermissionError:
                                print(f"Error: Target file {new_filepath} is open. Cannot overwrite.")
                                continue
                        
                        try:
                            shutil.move(latest_file, new_filepath)
                            print(f"Success! Saved result to: {new_filename}")
                        except Exception as e:
                            print(f"Error moving file: {e}")
                    else:
                        print("Error: No output file found.")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Restore original configuration
        print("\nRestoring original configuration...")
        save_config(backup_config)
        print("Done.")

if __name__ == "__main__":
    main()
