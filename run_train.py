import traceback
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

try:
    with open(r"e:\EcoWatch-AI\train.py", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, "train.py", "exec"))
except Exception as e:
    print(f"\n\nERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
    with open(r"e:\EcoWatch-AI\train_error.txt", "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
