#!/usr/bin/env python
"""Auto-download cross-domain independent test data from public URLs."""
import urllib.request, zipfile, io, os, sys, json, time, logging, pandas as pd
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app, db
from app.models.dataset import Dataset
from app.models.user import User

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads', 'datasets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()
        except Exception as e:
            logger.warning('Attempt %d/3 failed: %s', attempt+1, e)
            if attempt < 2: time.sleep(2)
    return None

def csv_from_zip(content, pattern='.csv'):
    zf = zipfile.ZipFile(io.BytesIO(content))
    for f in zf.namelist():
        if f.endswith(pattern) or f.endswith('.data'):
            return zf.read(f).decode('utf-8', errors='replace')
    # fallback: try listing
    for f in zf.namelist():
        if not f.endswith('.names') and not f.endswith('.txt'):
            try:
                txt = zf.read(f).decode('utf-8', errors='replace')
                if len(txt) > 200: return txt
            except: pass
    logger.warning('Zip has: %s', zf.namelist())
    return None

SOURCES = [
    dict(tid=130, url='https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/weibo_senti_100k/weibo_senti_100k.csv', target='label', name='Weibo Sentiment 100k', note='Weibo vs Hotel reviews'),
    dict(tid=132, url='https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/online_shopping_10_cats/online_shopping_10_cats.csv', target='cat', name='Shopping 10 Cats (full)', note='Full categories from public mirror'),
    dict(tid=131, url='https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv', target='label', name='Waimai Reviews 10k', note='Food delivery vs movie reviews'),
    dict(tid=136, url='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip', target='label', name='SMS Spam', note='SMS spam vs email spam', parser='sms'),
]

def process_source(src, app_ctx):
    with app_ctx:
        training_ds = db.session.get(Dataset, src['tid'])
        if not training_ds:
            logger.warning('Dataset %d not found', src['tid']); return 'fail'
        exist = Dataset.query.filter_by(source_dataset_id=training_ds.id, is_test_set=True, collection_method='url').first()
        if exist: logger.info('SKIP: %s', training_ds.name); return 'skip'
        logger.info('Download: %s', src['url'][:80])
        content = download(src['url'])
        if not content: logger.error('FAIL download'); return 'fail'
        df = None
        if src.get('parser') == 'sms':
            txt = csv_from_zip(content, '.txt') or content.decode('utf-8','replace')
            rows = []
            for line in txt.strip().split(chr(10)):
                if chr(9) in line:
                    label, msg = line.split(chr(9), 1)
                    rows.append({'label': label.strip(), 'message': msg.strip()})
            if rows: df = pd.DataFrame(rows)
        else:
            try: df = pd.read_csv(io.StringIO(content.decode('utf-8','replace')))
            except: pass
        if df is None or len(df) < 20: logger.error('FAIL parse (rows=%d)', len(df) if df is not None else 0); return 'fail'
        if len(df) > 10000: df = df.sample(10000, random_state=42)
        tc = src['target']
        if tc not in df.columns: tc = df.columns[-1]
        out_name = 'indtest_%s.csv' % src['name'].replace(' ','_').lower()[:40]
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df.to_csv(out_path, index=False)
        admin = User.query.filter_by(username='admin').first() or User.query.first()
        test_ds = Dataset(
            name='%s [IndTest-url]' % training_ds.name,
            description=src.get('note',''),
            file_path=out_path, file_size=os.path.getsize(out_path), file_format='csv',
            category=training_ds.category, row_count=len(df), column_count=len(df.columns),
            summary_json=json.dumps({'target_column': tc, 'n_samples': len(df), 'source_url': src['url'][:200], 'note': src.get('note','')}, ensure_ascii=False),
            status='ready', is_public=False, is_test_set=True,
            source_dataset_id=training_ds.id, collection_method='url',
            owner_id=admin.id if admin else 1,
        )
        db.session.add(test_ds); db.session.commit()
        logger.info('OK: %s (%d rows)', test_ds.name, len(df))
        return 'ok'

if __name__ == '__main__':
    app = create_app()
    with app.app_context(): pass  # init
    stats = {'ok':0,'skip':0,'fail':0}
    for src in SOURCES:
        r = process_source(src, app.app_context())
        stats[r] = stats.get(r,0) + 1
    logger.info('Done: ok=%d skip=%d fail=%d', stats['ok'], stats['skip'], stats['fail'])
