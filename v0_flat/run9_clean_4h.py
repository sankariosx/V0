import json
from pathlib import Path
import numpy as np,pandas as pd
from features import build_features
from validation import split_discovery_validation,block_wild_null_test
R=Path(__file__).resolve().parent; df=pd.read_csv(R/"data/xauusd_m15.csv"); df.columns=[c.lower() for c in df.columns]
ts="date" if "date" in df.columns else "datetime" if "datetime" in df.columns else "timestamp"
df[ts]=pd.to_datetime(df[ts],unit="ms",utc=True) if ts=="timestamp" and pd.api.types.is_numeric_dtype(df[ts]) else pd.to_datetime(df[ts],utc=True)
df=df.set_index(ts).sort_index()
feat,target=build_features(df,horizon=16,vol_window=32); idx=pd.DatetimeIndex(df.index)
future=pd.Series(idx,index=idx).shift(-16); valid=((future-pd.Series(idx,index=idx))==pd.Timedelta(minutes=240)).reindex(target.index).fillna(False).to_numpy()
feat,target=feat.loc[valid],target.loc[valid]; raw=np.log(df.close.shift(-16)/df.close).loc[feat.index]
_,_,fv,yv=split_discovery_validation(feat,target,frac=.60,purge=16); raw=raw.loc[yv.index]; st=pd.DatetimeIndex(yv.index)
rules={"dist_low_20+hour_cos":[("dist_low_20","<",0.0003755636445740603),("hour_cos",">",0.7933533402912352)],"range_+hour_cos":[("range_","<",0.00014818282204814845),("hour_cos",">",0.7933533402912352)],"rel_pos_50+hour_cos":[("rel_pos_50","<",0.15929677521127186),("hour_cos",">",0.7933533402912352)]}
def cond(f,r):
 m=np.ones(len(f),bool)
 for k,o,t in r:
  x=f[k].to_numpy(); m &= x<t if o=="<" else x>t
 return m
out={"rows":len(yv),"start":str(st[0]),"end":str(st[-1])}
for n,r in rules.items():
 c=cond(fv,r); a=c&(st.minute==0)&(st.hour%4==0)&(st.dayofweek<5)
 z=raw.to_numpy()[a]; y=yv.to_numpy()[a]
 out[n]={"n":int(a.sum()),"zero":int((z==0).sum()),"mean_raw":float(z.mean()) if len(z) else 0,"median_raw":float(np.median(z)) if len(z) else 0,"hit":float((z>0).mean()) if len(z) else 0,"effect_norm":float(y.mean()-yv.to_numpy().mean()) if len(y) else 0,"p":float(block_wild_null_test(yv.to_numpy(),a,block_len=64,n_perm=10000,seed=20261004)["p_value"]),"late_n":int(a[len(a)//2:].sum()),"late_mean":float(z[len(z)//2:].mean()) if len(z)>1 else 0}
(R/"results/run9_clean_4h.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
