import json
from pathlib import Path
import numpy as np, pandas as pd
from features import build_features
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"data/xauusd_m15.csv"); df.columns=[c.lower() for c in df.columns]
ts="date" if "date" in df.columns else "datetime" if "datetime" in df.columns else "timestamp"
df[ts]=pd.to_datetime(df[ts],unit="ms",utc=True) if ts=="timestamp" and pd.api.types.is_numeric_dtype(df[ts]) else pd.to_datetime(df[ts],utc=True)
df=df.set_index(ts).sort_index()
feat,target=build_features(df,horizon=16,vol_window=32)
idx=pd.DatetimeIndex(df.index)
future=pd.Series(idx,index=idx).shift(-16)
valid=((future-pd.Series(idx,index=idx))==pd.Timedelta(minutes=240)).reindex(target.index).fillna(False).to_numpy()
feat,target=feat.loc[valid],target.loc[valid]
raw=np.log(df["close"].shift(-16)/df["close"]).loc[feat.index]
_,_,fv,yv=__import__("validation").split_discovery_validation(feat,target,frac=.60,purge=16)
raw=raw.loc[yv.index]
rules={
"dist_low_20+hour_cos":[("dist_low_20","<",0.0003755636445740603),("hour_cos",">",0.7933533402912352)],
"range_+hour_cos":[("range_","<",0.00014818282204814845),("hour_cos",">",0.7933533402912352)],
"rel_pos_50+hour_cos":[("rel_pos_50","<",0.15929677521127186),("hour_cos",">",0.7933533402912352)]}
def cond(f,r):
 m=np.ones(len(f),bool)
 for k,o,t in r:
  x=f[k].to_numpy(); m &= x<t if o=="<" else x>t
 return m
out={"rows":len(yv),"validation_start":str(yv.index[0]),"validation_end":str(yv.index[-1]),"duplicate_timestamps":int(df.index.duplicated().sum()),"exact_zero_raw_all":int((raw==0).sum())}
for name,r in rules.items():
 c=cond(fv,r)
 out[name]={}
 for off in range(4):
  a=(fv.index.minute==0)&(((fv.index.hour-off)%4)==0)&c
  z=raw.to_numpy()[a]
  out[name][f"offset_{off}"]={"n":int(a.sum()),"zero":int((z==0).sum()),"mean":float(z.mean()) if len(z) else 0,"median":float(np.median(z)) if len(z) else 0,"hit":float((z>0).mean()) if len(z) else 0,"q10":float(np.quantile(z,.1)) if len(z) else 0,"q90":float(np.quantile(z,.9)) if len(z) else 0}
out["hour_cos_boundary_check"]={}
for h in [0,1,2,22,23]:
 a=(fv.index.hour==h)&(fv.index.minute==0)
 z=raw.to_numpy()[a]
 out["hour_cos_boundary_check"][str(h)]={"n":int(a.sum()),"zero":int((z==0).sum()),"mean":float(z.mean()) if len(z) else 0,"hit":float((z>0).mean()) if len(z) else 0}
(ROOT/"results/run8_4h_integrity.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
