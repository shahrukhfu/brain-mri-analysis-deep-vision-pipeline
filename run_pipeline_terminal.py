import json
import sys
import os

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch_lib = r"c:\Users\shahr\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except:
        pass

# Mock plt.show to prevent terminal hangs while saving figures to disk
import matplotlib
matplotlib.use('Agg')

print("Loading notebook mri_analysis_pipeline.ipynb...")
with open('mri_analysis_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Starting execution of notebook cells in terminal...")
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = "".join(cell['source'])
        print(f"\n==========================================")
        print(f"Executing Code Cell {idx}...")
        print(f"==========================================")
        try:
            exec(code, globals())
        except Exception as e:
            print(f"\nError in Cell {idx}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

print("\n==========================================")
print("SUCCESS: All notebook cells executed perfectly in the terminal!")
print("All comparative figures and results are saved in the 'before_after_dataset' folder.")
print("==========================================")
