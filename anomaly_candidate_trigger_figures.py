"""Render field-only trigger diagnostics for candidate episodes.

Each SVG shows the trigger sensor's actual 5-minute mean, actual trigger metric, and robust
score against the frozen bundle threshold. Actual values must not leave the field.
"""
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_contract import AnomalyContractError, canonical_sha256, ensure_output_directory, require, require_columns, sha256_file, write_json
from anomaly_validated_workflow import M0_BUNDLE_CONTRACT

CONTRACT="PIMS-ANOMALY-CANDIDATE-TRIGGER-FIGURE-v1"
BASE_COLUMNS=["window_start","window_end","asset_id","sensor_uid","measurement_type","unit","quality_status","mean"]


def _statsmodels()->Any:
    try: import statsmodels.api as sm
    except ImportError as exc: raise AnomalyContractError("STATSMODELS_REQUIRED","Approved field Statsmodels is required") from exc
    return sm


def _bundle(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8")); require(value.get("contract_version")==M0_BUNDLE_CONTRACT,"M0_BUNDLE_VERSION_INVALID",str(path)); return value


def collect(candidates:pd.DataFrame,feature_paths:list[Path],context_hours:float,chunksize:int)->tuple[pd.DataFrame,int]:
    requests=[]
    for row in candidates.itertuples():
        trigger=str(row.top_trigger_1); require("::" in trigger,"TOP_TRIGGER_INVALID",str(row.candidate_uid)); sensor,metric=trigger.split("::",1); requests.append((str(row.candidate_uid),str(row.asset_id),sensor,metric,pd.Timestamp(row.episode_start).tz_convert("UTC")-pd.Timedelta(hours=context_hours),pd.Timestamp(row.episode_end).tz_convert("UTC")+pd.Timedelta(hours=context_hours)))
    selected=[]; scanned=0; metrics=sorted({item[3] for item in requests})
    for source in feature_paths:
        header=pd.read_csv(source,nrows=0).columns.tolist(); require_columns(header,BASE_COLUMNS,"TRIGGER_FIGURE_FEATURE_HEADER_INVALID"); usecols=[column for column in [*BASE_COLUMNS,"min","max",*metrics] if column in header]
        for chunk in pd.read_csv(source,usecols=usecols,chunksize=chunksize,low_memory=False):
            scanned+=len(chunk); times=pd.to_datetime(chunk["window_start"],utc=True,errors="coerce"); require(not times.isna().any(),"TRIGGER_FIGURE_TIME_INVALID",str(source))
            for uid,asset,sensor,metric,start,end in requests:
                require(metric in chunk.columns,"TRIGGER_METRIC_MISSING",metric); mask=(chunk["asset_id"]==asset)&(chunk["sensor_uid"]==sensor)&(times>=start)&(times<end)
                if mask.any(): part=chunk.loc[mask].copy(); part.insert(0,"candidate_uid",uid); part["plot_time_utc"]=times.loc[mask]; part["trigger_metric_name"]=metric; selected.append(part)
    require(bool(selected),"TRIGGER_FIGURE_DATA_EMPTY","No trigger sensor rows matched"); return pd.concat(selected,ignore_index=True),scanned


def _render(candidate:pd.Series,data:pd.DataFrame,model:dict[str,Any],path:Path,context_hours:float,lowess_fraction:float)->None:
    sm=_statsmodels(); sensor,metric=str(candidate["top_trigger_1"]).split("::",1); parameter=next((item for item in model["feature_parameters"] if item["sensor_uid"]==sensor and item["metric"]==metric),None); require(parameter is not None,"TRIGGER_PARAMETER_MISSING",f"{sensor}:{metric}")
    data=data.sort_values("plot_time_utc"); data=data[data["quality_status"]=="PASS"].copy(); data["mean_value"]=pd.to_numeric(data["mean"],errors="coerce"); data["metric_value"]=pd.to_numeric(data[metric],errors="coerce"); data["robust_score"]=(data["metric_value"]-float(parameter["center"])).abs()/float(parameter["robust_scale"])
    start=pd.Timestamp(candidate["episode_start"]).tz_convert("UTC")-pd.Timedelta(hours=context_hours); end=pd.Timestamp(candidate["episode_end"]).tz_convert("UTC")+pd.Timedelta(hours=context_hours); ep_start=pd.Timestamp(candidate["episode_start"]).tz_convert("UTC"); ep_end=pd.Timestamp(candidate["episode_end"]).tz_convert("UTC")
    width,left,right,top,panel_h,gap=1600,150,55,125,220,35; specs=[("Actual 5-minute sensor mean","mean_value",str(data.iloc[0]["unit"]),None),(f"Actual trigger metric: {metric}","metric_value",str(data.iloc[0]["unit"]),None),("Robust trigger score","robust_score","robust z",float(model["asset_threshold"]))]; height=top+len(specs)*(panel_h+gap)+55
    def xpos(ts:pd.Timestamp)->float:return left+(ts.value-start.value)/max(end.value-start.value,1)*(width-left-right)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,"Malgun Gothic","Noto Sans KR",sans-serif;fill:#172033}.grid{stroke:#e2e8f0;stroke-width:1}.actual{fill:none;stroke:#64748b;stroke-width:1}.smooth{fill:none;stroke:#2563eb;stroke-width:2}.small{font-size:11px}</style>',f'<text x="{left}" y="30" font-size="20" font-weight="bold">{escape(str(candidate["candidate_uid"]))} — Trigger Diagnostic</text>',f'<text x="{left}" y="55" font-size="13">{escape(str(candidate["asset_id"]))} · {escape(sensor)} · {escape(metric)}</text>',f'<text x="{left}" y="78" font-size="12" fill="#b91c1c">현장 내부 실제 값 · 빨강 음영: candidate · 파랑: LOWESS · score/threshold 값 외부 반출 금지</text>']
    for index,(title,column,unit,threshold) in enumerate(specs):
        panel_top=top+index*(panel_h+gap); plot_top,plot_bottom=panel_top+28,panel_top+panel_h-35; values=data[["plot_time_utc",column]].dropna(); svg.extend([f'<rect x="{left}" y="{plot_top}" width="{width-left-right}" height="{plot_bottom-plot_top}" fill="#fff" stroke="#cbd5e1"/>',f'<text x="{left}" y="{panel_top+17}" font-size="13" font-weight="bold">{escape(title)} [{escape(unit)}]</text>']); x1,x2=xpos(ep_start),xpos(ep_end); svg.append(f'<rect x="{x1:.2f}" y="{plot_top}" width="{max(x2-x1,2):.2f}" height="{plot_bottom-plot_top}" fill="#ef4444" opacity="0.14"/>')
        if values.empty: continue
        y_min,y_max=float(values[column].min()),float(values[column].max());
        if threshold is not None: y_min=min(y_min,threshold); y_max=max(y_max,threshold)
        pad=max((y_max-y_min)*.08,abs(y_max)*.001,1e-9); y_min-=pad; y_max+=pad
        def ypos(value:float)->float:return plot_bottom-(value-y_min)/max(y_max-y_min,1e-12)*(plot_bottom-plot_top)
        for tick in range(5):
            y=plot_top+tick*(plot_bottom-plot_top)/4; label=y_max-tick*(y_max-y_min)/4; svg.extend([f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>',f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" class="small">{label:.5g}</text>'])
        points=" ".join(f'{xpos(row.plot_time_utc):.2f},{ypos(float(getattr(row,column))):.2f}' for row in values.itertuples());
        if len(values)>1: svg.append(f'<polyline points="{points}" class="actual"/>')
        if len(values)>=8:
            smooth=sm.nonparametric.lowess(values[column].to_numpy(),np.arange(len(values)),frac=lowess_fraction,return_sorted=False); smooth_points=" ".join(f'{xpos(ts):.2f},{ypos(float(value)):.2f}' for ts,value in zip(values["plot_time_utc"],smooth)); svg.append(f'<polyline points="{smooth_points}" class="smooth"/>')
        if threshold is not None:
            y=ypos(threshold); svg.extend([f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="7 4"/>',f'<text x="{width-right-5}" y="{y-5:.2f}" text-anchor="end" class="small">frozen asset threshold</text>'])
        for tick in range(7):
            ts=start+(end-start)*tick/6; x=xpos(ts); svg.append(f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" class="grid"/>')
            if index==len(specs)-1: svg.append(f'<text x="{x:.2f}" y="{plot_bottom+19}" text-anchor="middle" class="small">{ts.strftime("%m-%d %H:%M")}</text>')
    svg.extend([f'<text x="{left}" y="{height-18}" font-size="12">max_abs_change는 5분 내 유효 1초 표본의 인접값 최대 절대변화다. Spike/전환/센서 artifact를 1초 zoom으로 확인.</text>','</svg>']); path.write_text("\n".join(svg)+"\n",encoding="utf-8")


def build(candidates_path:Path,bundle_path:Path,feature_paths:list[Path],output_dir:Path,context_hours:float,lowess_fraction:float,chunksize:int,overwrite:bool)->dict[str,Any]:
    output=ensure_output_directory(output_dir,overwrite); candidates=pd.read_csv(candidates_path,dtype=str).fillna(""); require_columns(candidates.columns,["candidate_uid","asset_id","episode_start","episode_end","top_trigger_1"],"CANDIDATE_INDEX_INVALID"); bundle=_bundle(bundle_path); models={model["asset_id"]:model for model in bundle["models"]}; data,scanned=collect(candidates,feature_paths,context_hours,chunksize); data_path=output/"trigger_plot_data.csv"; data.to_csv(data_path,index=False); figure_dir=output/"figures"; figure_dir.mkdir(); index=[]
    for _,candidate in candidates.iterrows():
        path=figure_dir/f'{candidate["candidate_uid"]}_trigger_diagnostic.svg'; _render(candidate,data[data["candidate_uid"]==candidate["candidate_uid"]],models[candidate["asset_id"]],path,context_hours,lowess_fraction); index.append({"candidate_uid":candidate["candidate_uid"],"asset_id":candidate["asset_id"],"top_trigger":candidate["top_trigger_1"],"figure_file":path.name,"field_only":True})
    index_frame=pd.DataFrame(index); index_path=output/"trigger_figure_index.csv"; index_frame.to_csv(index_path,index=False); lines=[f"VC20A|{asset}|EP:{len(group)}|FIG:{len(group)}|TRIGGERS:{group['top_trigger'].nunique()}" for asset,group in index_frame.groupby("asset_id",sort=True)]; payload={"contract_version":CONTRACT,"bundle_id":bundle["model_bundle_id"],"candidate_source_sha256":sha256_file(candidates_path),"bundle_file_sha256":sha256_file(bundle_path),"feature_source_sha256":[sha256_file(path) for path in feature_paths],"scanned_rows":scanned,"selected_rows":len(data),"figure_count":len(index_frame),"detail_lines":lines,"outputs":{"data_sha256":sha256_file(data_path),"index_sha256":sha256_file(index_path)},"safety":{"actual_values_present":True,"field_only":True,"external_transfer_allowed":False,"production_use_allowed":False}}
    sha=canonical_sha256(payload); manifest={**payload,"receipt_sha256":sha}; write_json(output/"vc20_manifest.json",manifest); receipt=f"VC20=VALID|EP:{len(index_frame)}|FIG:{len(index_frame)}|SCANNED:{scanned}|ROWS:{len(data)}|SHA:{sha}"; (output/"vc20_receipt.txt").write_text(receipt+"\n",encoding="utf-8"); (output/"vc20_compact_detail.txt").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(receipt); print("\n".join(lines)); return manifest


def parser()->argparse.ArgumentParser:
    value=argparse.ArgumentParser(description=__doc__); value.add_argument("--candidates",type=Path,required=True); value.add_argument("--bundle",type=Path,required=True); value.add_argument("--features",type=Path,nargs="+",required=True); value.add_argument("--output",type=Path,required=True); value.add_argument("--context-hours",type=float,default=6); value.add_argument("--lowess-fraction",type=float,default=.1); value.add_argument("--chunksize",type=int,default=200000); value.add_argument("--overwrite",action="store_true"); return value


def main()->int:
    args=parser().parse_args(); build(args.candidates,args.bundle,args.features,args.output,args.context_hours,args.lowess_fraction,args.chunksize,args.overwrite); return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except AnomalyContractError as exc:
        print(f"TRIGGER_FIGURE=BLOCKED|CODE:{exc.code}"); raise SystemExit(2) from None
