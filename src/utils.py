from pathlib import Path
import re

def norm(c):
    return re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_")

def day_of(path):
    """Dataset files use DD-MM-YYYY; CSV legacy filenames use MM-DD-YYYY."""
    import re
    name = Path(path).name
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", name)
    if not match:
        raise ValueError(f"Không tìm thấy ngày: {name}")
    a, b, y = map(int, match.groups())
    if name.lower().endswith(".csv") and not any(x in name.lower() for x in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
        month, day = a, b
    else:
        day, month = a, b
    from datetime import date
    return date(y, month, day).isoformat()

def log(msg):
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
