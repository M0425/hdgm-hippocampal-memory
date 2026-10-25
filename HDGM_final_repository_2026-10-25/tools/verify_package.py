from pathlib import Path
import csv,hashlib,sys
r=Path(__file__).resolve().parents[1]
bad=[]
for row in csv.DictReader((r/'MANIFEST.csv').open(encoding='utf-8')):
 p=r/row['file']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:bad.append(row['file'])
print('Manifest failures:',bad)
sys.exit(bool(bad))
