"""Synthetic demonstration only. No real claims or independent human labels."""
from pathlib import Path
import csv, hashlib, json, platform, time
from datetime import datetime, timedelta, timezone
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ['damage', 'delay', 'shortage']

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def f1(y, pred):
    scores = []
    for k in range(3):
        tp = np.sum((y == k) & (pred == k))
        fp = np.sum((y != k) & (pred == k))
        fn = np.sum((y == k) & (pred != k))
        scores.append(float(2*tp / max(2*tp+fp+fn, 1)))
    return float(np.mean(scores)), scores

def probabilities(x, w):
    z = x @ w; z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); return p / p.sum(axis=1, keepdims=True)

def fit(x, y, reg, lr, epochs):
    w = np.zeros((x.shape[1], 3)); target = np.eye(3)[y]
    for _ in range(epochs):
        penalty = reg*w; penalty[0] = 0
        w -= lr * (x.T @ (probabilities(x,w)-target)/len(y)+penalty)
    return w

def main():
    started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter()
    cfg = json.loads((ROOT/'config.json').read_text())
    rng = np.random.default_rng(cfg['generation_seed'])
    n = cfg['n_claims']; rows = []; events = []
    for i in range(n):
        y = int(rng.choice(3,p=cfg['class_probabilities']))
        active = [y]
        if rng.random() < cfg['multi_cause_probability']:
            active.append(int(rng.choice([k for k in range(3) if k != y])))
        region = str(rng.choice(['North America','Europe','Asia Pacific','Latin America']))
        carrier = int(rng.integers(1,81)); quality = float(rng.uniform(.45,1))
        sla = float(rng.choice([2,4,8,12,24])); quantity = int(rng.integers(20,501))
        allowed_loss = float(rng.choice([0,.01,.03]))
        image = float(np.clip(rng.normal(.7 if 0 in active else .25,.24),0,1))
        delay = float(max(0,rng.normal(1.65 if 1 in active else .6,.65))*sla)
        loss = float(np.clip(rng.normal(.10 if 2 in active else .015,.035),0,.5))
        actual_qty = max(0,round(quantity*(1-loss)))
        extracted_qty = max(0,actual_qty+int(round(rng.normal(0,(1-quality)*12))))
        value = round(float(rng.lognormal(6.1,1.05)),2)
        stamp = datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(seconds=i*365*86400/n)
        split = 'train' if i < int(n*.67) else 'validation' if i < int(n*.83) else 'test'
        row = dict(provenance='SYNTHETIC_NOT_REAL',claim_id=f'SYN-{i+1:06d}',
                   freight_order_id=f'SYN-FO-{i+1:06d}',event_timestamp=stamp.isoformat(),split=split,
                   carrier_id=f'SYN-C{carrier:03d}',region=region,claim_value_usd=value,
                   contract_payment_days=int(rng.choice([15,30,45,60])),document_quality=round(quality,4),
                   image_damage_proxy=round(image,4),delay_hours=round(delay,3),sla_buffer_hours=sla,
                   shipped_quantity=quantity,extracted_received_quantity=extracted_qty,
                   contractual_loss_tolerance=allowed_loss,
                   synthetic_primary_label=LABELS[y],synthetic_possible_labels='|'.join(LABELS[k] for k in sorted(active)),
                   synthetic_ambiguous=len(active)>1,
                   evidence_text=f'SIMULATED: transit deviation {delay:.1f} h; load {quantity} pieces; received-document reading {extracted_qty}; image damage proxy {image:.2f}.')
        rows.append(row)
        events.append(dict(provenance='SYNTHETIC_NOT_REAL',event_type='simulated_claim_received',
                           event_timestamp=stamp.isoformat(),claim_id=row['claim_id'],payload=row))
    save_csv(ROOT/'data/synthetic_claims.csv', rows)
    (ROOT/'data/simulated_events.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')
    # Derive features without reading any label, split, carrier ID, or timestamp.
    raw = np.array([[r['image_damage_proxy'],r['delay_hours'],
                     r['shipped_quantity']-r['extracted_received_quantity'],r['document_quality']] for r in rows])
    grounded = np.array([[r['image_damage_proxy'],r['delay_hours']/r['sla_buffer_hours'],
                         (r['shipped_quantity']-r['extracted_received_quantity'])/r['shipped_quantity']-r['contractual_loss_tolerance'],
                         r['document_quality']] for r in rows])
    y = np.array([LABELS.index(r['synthetic_primary_label']) for r in rows])
    masks = {s:np.array([r['split']==s for r in rows]) for s in ['train','validation','test']}
    assert max(r['event_timestamp'] for r in rows if r['split']=='train') < min(r['event_timestamp'] for r in rows if r['split']=='validation')
    assert len({r['freight_order_id'] for r in rows}) == n
    metrics={}; trials=[]; selected={}; predictions=[]
    models = {'D1_raw_proxies':raw,'D2_contract_normalized_proxies':grounded,'D3_without_image_proxy':grounded[:,1:]}
    for name, features in models.items():
        mean=features[masks['train']].mean(axis=0); sd=features[masks['train']].std(axis=0); sd[sd==0]=1
        x=np.column_stack([np.ones(n),(features-mean)/sd]); best=None
        for reg in cfg['search']['l2']:
            for lr in cfg['search']['learning_rate']:
                w=fit(x[masks['train']],y[masks['train']],reg,lr,cfg['epochs'])
                score,_=f1(y[masks['validation']],probabilities(x[masks['validation']],w).argmax(axis=1))
                trials.append(dict(model=name,l2=reg,learning_rate=lr,epochs=cfg['epochs'],validation_macro_f1=score,provenance='ACTUAL_RUN_ON_SYNTHETIC_DATA'))
                if best is None or score>best[0]: best=(score,w,reg,lr)
        score,w,reg,lr=best
        selected[name]=dict(l2=reg,learning_rate=lr,epochs=cfg['epochs'],validation_macro_f1=score,features=list(range(features.shape[1])))
        model_dir=ROOT/'models';model_dir.mkdir(exist_ok=True)
        np.savez(model_dir/(name+'.npz'),weights=w,mean=mean,std=sd,labels=np.array(LABELS))
        p=probabilities(x[masks['test']],w); pred=p.argmax(axis=1); truth=y[masks['test']]
        macro,per_class=f1(truth,pred);test_rows=[r for r in rows if r['split']=='test']
        boot_rng=np.random.default_rng(cfg['bootstrap_seed']);boot=[]
        for _ in range(cfg['bootstrap_resamples']):
            idx=boot_rng.integers(0,len(truth),len(truth));boot.append(f1(truth[idx],pred[idx])[0])
        gate=p.max(axis=1)>=cfg['confidence_threshold']
        metrics[name]=dict(n_test=len(truth),macro_f1=macro,per_class_f1=dict(zip(LABELS,per_class)),
            bootstrap_95_percent_interval=np.quantile(boot,[.025,.975]).tolist(),accuracy=float(np.mean(pred==truth)),
            classification_confidence_coverage=float(gate.mean()),
            accuracy_above_threshold=float(np.mean(pred[gate]==truth[gate])) if gate.any() else None,
            ambiguity_aware_accuracy=float(np.mean([LABELS[k] in r['synthetic_possible_labels'].split('|') for k,r in zip(pred,test_rows)])))
        for r,pr,k in zip(test_rows,p,pred):
            predictions.append(dict(provenance='ACTUAL_PREDICTION_ON_SYNTHETIC_DATA',model=name,claim_id=r['claim_id'],
                synthetic_truth=r['synthetic_primary_label'],prediction=LABELS[k],
                probability_damage=float(pr[0]),probability_delay=float(pr[1]),probability_shortage=float(pr[2]),
                passes_classification_threshold=bool(max(pr)>=cfg['confidence_threshold'])))
        subgroups=[]
        for field in ['region','synthetic_ambiguous']:
            for value in sorted({str(r[field]) for r in test_rows}):
                idx=np.array([str(r[field])==value for r in test_rows])
                subgroups.append(dict(group_field=field,group_value=value,n=int(idx.sum()),macro_f1=f1(truth[idx],pred[idx])[0]))
        metrics[name]['subgroups']=subgroups
    save_csv(ROOT/'results/test_predictions.csv',predictions)
    save_csv(ROOT/'logs/validation_trials.csv',trials)
    save_json(ROOT/'results/metrics.json',metrics);save_json(ROOT/'results/selected_configurations.json',selected)
    # Human annotation fields intentionally remain empty; exclude generator truth and predictions.
    candidates=[r for r in rows if r['split']=='test']
    chosen=np.random.default_rng(cfg['annotation_sampling_seed']).choice(len(candidates),size=150,replace=False)
    packet=[]; forms=[]
    for j in chosen:
        r=candidates[int(j)]
        packet.append({k:v for k,v in r.items() if not k.startswith('synthetic_') and k!='split'})
        for slot in range(1,4):
            forms.append(dict(provenance='BLANK_HUMAN_ANNOTATION_FORM_FOR_SYNTHETIC_CASE',claim_id=r['claim_id'],reviewer_slot=slot,
                reviewer_id='',primary_cause='',additional_causes='',confidence='',evidence_rationale='',completed_at='',status='NOT_ANNOTATED'))
    save_csv(ROOT/'annotations/blinded_synthetic_evidence.csv',packet)
    save_csv(ROOT/'annotations/independent_annotation_form_BLANK.csv',forms)
    save_json(ROOT/'logs/run_log.json',dict(provenance='ACTUAL_EXECUTION_ON_SYNTHETIC_DATA',started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-tick,
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        split_counts={s:int(m.sum()) for s,m in masks.items()},trials=len(trials),independent_annotations_completed=0))
    checks=dict(unique_claim_ids=len({r['claim_id'] for r in rows})==n,
        strict_temporal_order=True,probabilities_sum_to_one=all(abs(sum(r[k] for k in ['probability_damage','probability_delay','probability_shortage'])-1)<1e-10 for r in predictions),
        finite_metrics=all(np.isfinite(m['macro_f1']) for m in metrics.values()),
        blank_annotation_fields=all(r['reviewer_id']=='' and r['primary_cause']=='' for r in forms))
    assert all(checks.values());save_json(ROOT/'logs/validation_checks.json',checks)
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='sha256_manifest.json'}
    save_json(ROOT/'sha256_manifest.json',manifest)
    print(json.dumps({k:round(v['macro_f1'],6) for k,v in metrics.items()}))

if __name__=='__main__': main()
