"""Validate real production ledger/export data without altering the active run."""
import csv,gzip,json,math,sqlite3,tempfile,time
from pathlib import Path
from production_common import ROOT,atomic,connect,sha,utc
from production_sync import batch_export,consolidate

def main():
    run=Path((ROOT/'current_production_run.txt').read_text().strip());source=connect(run)
    deadline=time.time()+180
    while time.time()<deadline:
        pairs=source.execute("SELECT count(*) FROM (SELECT pair_no FROM documents WHERE doc_id IN (SELECT doc_id FROM tasks GROUP BY doc_id HAVING sum(status!='done')=0) GROUP BY pair_no HAVING count(*)=2)").fetchone()[0]
        if pairs:break
        time.sleep(5)
    assert pairs>0,'No complete pair available for finalization check yet'
    assert source.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    counts={r[0]:r[1] for r in source.execute('SELECT model,count(*) FROM tasks GROUP BY model')}
    assert counts=={'llada':133968,'llama2_7b':5582,'llama3_8b':5582,'mistral7b_v01':5582,'olmo7b':5582}
    assert source.execute("SELECT count(*) FROM tasks WHERE status='failed'").fetchone()[0]==0
    assert source.execute("SELECT count(*) FROM (SELECT input_id FROM tasks WHERE model='llada' GROUP BY input_id HAVING count(*)!=24)").fetchone()[0]==0
    parent=ROOT/'tests';parent.mkdir(exist_ok=True)
    isolated=Path(tempfile.mkdtemp(prefix='production-finalization-',dir=parent));(isolated/'export').mkdir()
    target=connect(isolated);source.backup(target);source.close()
    config=json.loads((run/'config.json').read_text());batch_export(isolated,target)
    report=consolidate(isolated,target,config)
    assert report['committed_document_pairs']>=1 and len(report['files'])==28
    common=None;total=0
    for entry in report['files']:
        file=isolated/'export'/entry['file'];rows=list(csv.DictReader(file.open()))
        assert len(rows)==entry['rows'] and sha(file)==entry['sha256']
        identities={(r['group'],r['filtered_index'],r['window_index']) for r in rows}
        assert len(identities)==len(rows)
        if common is None:common=identities
        else:assert identities==common,'Final model/mask CSVs cover different windows'
        for r in rows:
            p=float(r['p_z']);logp=float(r['log_p_z'])
            assert 0<=p<=1 and math.isfinite(p) and not math.isnan(logp) and logp<=0
            assert math.isclose(p,math.exp(logp),rel_tol=1e-8,abs_tol=5e-324)
        total+=len(rows)
    assert total==28*len(common)==report['committed_rows']
    out=dict(status='passed',verified_utc=utc(),isolated_test_path=str(isolated),
        source_ledger_integrity='ok',expected_counts_verified=counts,committed_pairs_in_test=report['committed_document_pairs'],
        final_csvs_checked=28,identical_window_sets_in_all_final_csvs=True,rows_checked=total,
        real_run_unmodified=True,no_test_data_uploaded=True)
    atomic(run/'export/finalization_validation.json',out)
    atomic(run/'finalization_validation.json',out)
    print('PRODUCTION_FINALIZATION_INTEGRITY_PASSED',json.dumps(out),flush=True)

if __name__=='__main__':main()
