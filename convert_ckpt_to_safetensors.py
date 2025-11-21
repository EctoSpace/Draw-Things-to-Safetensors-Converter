import sqlite3
import torch
from safetensors.torch import save_file
import struct
import os
import argparse
from pathlib import Path

def convert_ckpt_to_safetensors(ckpt_file, overwrite=False, remove_ckpt=False):
    """Convert a single Draw Things .ckpt file to .safetensors format."""
    print(f"\n{'='*60}")
    print(f"Converting: {ckpt_file}")
    print(f"{'='*60}")
    
    # Create output filename (same location as input)
    output_file = ckpt_file.replace(".ckpt", ".safetensors")
    
    # Check if output already exists
    if os.path.exists(output_file) and not overwrite:
        print(f"⊘ Skipping - output already exists (use --overwrite to replace)")
        return False
    
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
            
            # Clean up tensor name - remove Draw Things specific prefixes/suffixes
            # Draw Things might wrap names like "__text_model__[t-8-0]__up__"
            # We need to extract the actual parameter name
            clean_name = name
            if clean_name.startswith("__") and "__" in clean_name[2:]:
                # Extract the actual name between __ markers
                parts = clean_name.strip("_").split("__")
                # Look for the actual parameter name (usually the middle part)
                for part in parts:
                    if not part.startswith("[") and part:
                        clean_name = part
                        break
            
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
        
        # Save as safetensors with metadata
        print(f"Saving to {output_file}...")
        metadata = {"format": "pt"}  # PyTorch format metadata
        save_file(state_dict, output_file, metadata=metadata)
        print(f"✓ Successfully converted! Saved {len(state_dict)} tensors")
        
        # Remove original .ckpt file if requested
        if remove_ckpt:
            os.remove(ckpt_file)
            print(f"🗑️  Removed original .ckpt file")
        
        return True
        
    except Exception as e:
        print(f"✗ Error converting {ckpt_file}: {e}")
        return False

def find_ckpt_files(folder_path):
    """Recursively find all .ckpt files in folder and subfolders."""
    ckpt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files

def main():
    parser = argparse.ArgumentParser(
        description='Convert Draw Things .ckpt files to .safetensors format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all .ckpt files in a folder (including subfolders)
  python convert_ckpt_to_safetensors.py --folder "c:/models"
  
  # Convert a single file
  python convert_ckpt_to_safetensors.py --file "c:/models/my_lora.ckpt"
  
  # Convert with overwrite and remove original
  python convert_ckpt_to_safetensors.py --folder "c:/models" --overwrite --remove-ckpt
        """
    )
    
    # Create mutually exclusive group for --folder and --file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder', type=str, help='Folder to scan for .ckpt files (includes subfolders)')
    group.add_argument('--file', type=str, help='Single .ckpt file to convert')
    
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing .safetensors files')
    parser.add_argument('--remove-ckpt', action='store_true',
                       help='Remove original .ckpt file after successful conversion')
    
    args = parser.parse_args()
    
    print("Draw Things .ckpt to .safetensors Converter")
    print("=" * 60)
    
    # Determine files to convert
    if args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return
        
        print(f"Scanning folder: {args.folder}")
        files_to_convert = find_ckpt_files(args.folder)
        
        if not files_to_convert:
            print(f"No .ckpt files found in {args.folder}")
            return
        
        print(f"Found {len(files_to_convert)} .ckpt file(s)\n")
    
    else:  # args.file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        
        if not args.file.endswith('.ckpt'):
            print(f"Error: File must have .ckpt extension")
            return
        
        files_to_convert = [args.file]
    
    # Display options
    print(f"Options:")
    print(f"  Overwrite existing: {args.overwrite}")
    print(f"  Remove .ckpt after conversion: {args.remove_ckpt}")
    
    # Convert files
    successful = 0
    skipped = 0
    failed = 0
    
    for ckpt_file in files_to_convert:
        result = convert_ckpt_to_safetensors(ckpt_file, args.overwrite, args.remove_ckpt)
        if result:
            successful += 1
        elif os.path.exists(ckpt_file.replace(".ckpt", ".safetensors")):
            skipped += 1
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

if __name__ == "__main__":
    main()