# Draw Things to Safetensors Converter

A Python tool to convert Draw Things `.ckpt` model files back to standard `.safetensors` format for use with ComfyUI, Automatic1111, and other AI image generation tools.

## Background

Draw Things stores AI models in a proprietary SQLite-based `.ckpt` format that is incompatible with most other AI tools. This converter extracts the tensor data from Draw Things' database format and saves it as standard `.safetensors` files that can be used across different platforms.

## Features

- ✅ Batch convert all `.ckpt` files in a folder
- ✅ Convert single files on demand
- ✅ Configurable input/output folders via JSON config
- ✅ Automatically skip already-converted files
- ✅ Preserves tensor names, shapes, and data types
- ✅ Progress tracking for large conversions
- ✅ Error handling with detailed reporting

## Requirements

- Python 3.7+
- PyTorch
- safetensors

## Installation

1. Clone this repository:
```bash
git clone https://github.com/EctoSpace/Draw-Things-to-Safetensors-Converter.git
cd Draw-Things-to-Safetensors-Converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Usage

The converter uses command-line arguments for easy and flexible operation.

#### Convert All Files in a Folder

Scans the folder and all subfolders for `.ckpt` files and converts them in place:

```bash
python convert_ckpt_to_safetensors.py --folder "c:/models"
```

On macOS/Linux:
```bash
python convert_ckpt_to_safetensors.py --folder "/Users/username/Library/Containers/com.liuliu.draw-things/Data/Documents/Models"
```

#### Convert a Single File

Converts a specific file, saving the `.safetensors` in the same location:

```bash
python convert_ckpt_to_safetensors.py --file "c:/models/my_lora.ckpt"
```

### Command-Line Options

- **`--folder <path>`**: Convert all `.ckpt` files in folder (includes subfolders)
- **`--file <path>`**: Convert a single `.ckpt` file
- **`--overwrite`**: Overwrite existing `.safetensors` files (optional)
- **`--remove-ckpt`**: Remove original `.ckpt` file after successful conversion (optional)

**Note:** You must use either `--folder` OR `--file`, not both.

### Examples

**Convert all files in a folder:**
```bash
python convert_ckpt_to_safetensors.py --folder "c:/models"
```

**Convert a single file:**
```bash
python convert_ckpt_to_safetensors.py --file "c:/models/sdxl_lora.ckpt"
```

**Convert with overwrite:**
```bash
python convert_ckpt_to_safetensors.py --folder "c:/models" --overwrite
```

**Convert and remove original .ckpt files:**
```bash
python convert_ckpt_to_safetensors.py --folder "c:/models" --remove-ckpt
```

**Convert with both options:**
```bash
python convert_ckpt_to_safetensors.py --folder "c:/models" --overwrite --remove-ckpt
```

**Get help:**
```bash
python convert_ckpt_to_safetensors.py --help
```

## Draw Things Model Locations

Default Draw Things model paths on macOS:

- **LoRAs**: `~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models/`
- **Checkpoints**: Same directory as above

Use these paths in your `input_folder` configuration.

## Output

The converter will:
1. Scan for `.ckpt` files in the input folder
2. Extract tensor data from each SQLite database
3. Convert tensors to appropriate PyTorch dtypes (float32, float16, etc.)
4. Save as `.safetensors` files in the output folder
5. Display a summary of successful, skipped, and failed conversions

Example output:
```
Draw Things .ckpt to .safetensors Converter
============================================================

Configuration:
  Input folder: ./models
  Output folder: ./converted_safetensors
  Single file: None (convert all)
  Overwrite existing: False

Found 3 .ckpt file(s) to convert

============================================================
Converting: model_1.ckpt
============================================================
Found 2112 tensors
  Processed 100/2112 tensors...
  Processed 200/2112 tensors...
  ...
Saving to ./converted_safetensors/model_1.safetensors...
✓ Successfully converted! Saved 2112 tensors

============================================================
CONVERSION SUMMARY
============================================================
Total files: 3
✓ Successfully converted: 3
⊘ Skipped (already exists): 0
✗ Failed: 0

Converted files saved to: ./converted_safetensors
```

## Troubleshooting

### "No .ckpt files found"
- Check that your `input_folder` path is correct
- Ensure the files have `.ckpt` extension
- Verify you have read permissions for the folder

### "Error converting: [Errno 2] No such file or directory"
- The input file path is incorrect
- Check the `input_folder` setting in `converter_config.json`

### Import errors
- Make sure PyTorch and safetensors are installed: `pip install torch safetensors`
- Activate your virtual environment if using one

### Memory issues with large models
- The converter processes one tensor at a time to minimize memory usage
- If you still encounter issues, try converting files one at a time using the `single_file` option

## Technical Details

Draw Things stores model weights in SQLite databases with the following structure:
- **Table**: `tensors`
- **Columns**: `name`, `type`, `format`, `datatype`, `dim` (shape), `data` (tensor bytes)

The converter:
1. Reads the SQLite database
2. Extracts tensor metadata (name, dimensions, datatype)
3. Infers PyTorch dtype from bytes-per-element
4. Reconstructs tensors with correct shapes
5. Saves using the safetensors library

## Limitations

- Only works with Draw Things `.ckpt` files (SQLite format)
- Cannot convert standard PyTorch `.ckpt` files (use other tools for that)
- Requires enough disk space for both input and output files

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

- Draw Things app by Liu Liu
- Safetensors library by Hugging Face
- PyTorch team

## Disclaimer

This tool is for converting your own legally obtained models. Ensure you have the right to use and convert any models before doing so. The authors are not responsible for any misuse of this tool.