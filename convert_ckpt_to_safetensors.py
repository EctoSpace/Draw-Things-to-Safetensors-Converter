import sqlite3
import torch
from safetensors.torch import save_file
import struct
import os
import json
from pathlib import Path

# Configuration file name
CONFIG_FILE = "converter_config.json"

def load_config():
    """Load configuration from JSON file or create default config."""
    default_config = {
        "input_folder": ".",  # Current directory by default
        "output_folder": "./converted_safetensors",  # Output folder
        "single_file": None,  # Set to a specific filename to convert only that file
        "overwrite_existing": False  # Whether to overwrite existing .safetensors files
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**default_config, **config}
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration")
    else:
        # Create default config file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default config file: {CONFIG_FILE}")
    
    return default_config

def convert_ckpt_to_safetensors(ckpt_file, output_folder):
    """Convert a single Draw Things .ckpt file to .safetensors format."""
    print(f"\n{'='*60}")
    print(f"Converting: {ckpt_file}")
    print(f"{'='*60}")
    
    # Create output filename
    base_name = Path(ckpt_file).stem
    output_file = os.path.join(output_folder, f"{base_name}.safetensors")
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(ckpt_file)
        cursor = conn.cursor()
        
        # Get all tensors
        cursor.execute("SELECT name, type, format, datatype, dim, data FROM tensors")
        rows = cursor.fetchall()
        
        print(f"Found {len(rows)} tensors")
        
        state_dict = {}
        
        for idx, row in enumerate(rows):
            name, tensor_type, format_type, datatype, dim_blob, data_blob = row
            
            # Parse dimensions from blob - remove trailing zeros
            if dim_blob:
                all_dims = struct.unpack(f'{len(dim_blob)//4}i', dim_blob)
                dims = tuple(d for d in all_dims if d > 0)
            else:
                dims = ()
            
            # Infer PyTorch dtype from data size and dimensions
            if dims:
                expected_elements = 1
                for d in dims:
                    expected_elements *= d
                bytes_per_element = len(data_blob) / expected_elements
                
                if bytes_per_element == 4:
                    dtype = torch.float32
                elif bytes_per_element == 2:
                    dtype = torch.float16
                elif bytes_per_element == 1:
                    dtype = torch.uint8
                else:
                    dtype = torch.float32
            else:
                dtype = torch.float32
            
            # Convert blob to tensor
            tensor_data = bytes(data_blob)
            tensor = torch.frombuffer(tensor_data, dtype=dtype).clone()
            
            if dims:
                tensor = tensor.reshape(dims)
            
            state_dict[name] = tensor
            
            # Show progress every 100 tensors
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(rows)} tensors...")
        
        conn.close()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Save as safetensors
        print(f"Saving to {output_file}...")
        save_file(state_dict, output_file)
        print(f"✓ Successfully converted! Saved {len(state_dict)} tensors")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {ckpt_file}: {e}")
        return False

def main():
    print("Draw Things .ckpt to .safetensors Converter")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"\nConfiguration:")
    print(f"  Input folder: {config['input_folder']}")
    print(f"  Output folder: {config['output_folder']}")
    print(f"  Single file: {config['single_file'] or 'None (convert all)'}")
    print(f"  Overwrite existing: {config['overwrite_existing']}")
    
    input_folder = config['input_folder']
    output_folder = config['output_folder']
    single_file = config['single_file']
    overwrite = config['overwrite_existing']
    
    # Determine which files to convert
    if single_file:
        # Convert only the specified file
        ckpt_path = os.path.join(input_folder, single_file)
        if not os.path.exists(ckpt_path):
            print(f"\nError: File not found: {ckpt_path}")
            return
        files_to_convert = [ckpt_path]
    else:
        # Find all .ckpt files in input folder
        files_to_convert = [
            os.path.join(input_folder, f) 
            for f in os.listdir(input_folder) 
            if f.endswith('.ckpt')
        ]
    
    if not files_to_convert:
        print(f"\nNo .ckpt files found in {input_folder}")
        return
    
    print(f"\nFound {len(files_to_convert)} .ckpt file(s) to convert")
    
    # Convert each file
    successful = 0
    skipped = 0
    failed = 0
    
    for ckpt_file in files_to_convert:
        base_name = Path(ckpt_file).stem
        output_file = os.path.join(output_folder, f"{base_name}.safetensors")
        
        # Check if output already exists
        if os.path.exists(output_file) and not overwrite:
            print(f"\nSkipping {ckpt_file} (output already exists)")
            skipped += 1
            continue
        
        if convert_ckpt_to_safetensors(ckpt_file, output_folder):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(files_to_convert)}")
    print(f"✓ Successfully converted: {successful}")
    print(f"⊘ Skipped (already exists): {skipped}")
    print(f"✗ Failed: {failed}")
    print(f"\nConverted files saved to: {output_folder}")

if __name__ == "__main__":
    main()