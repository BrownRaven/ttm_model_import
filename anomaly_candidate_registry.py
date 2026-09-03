"""Freeze field-confirmed PoC candidate episodes into a non-causal read-only registry."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_contract import AnomalyContractError, canonical_sha256, ensure_output_directory, require, require_columns, sha256_file, write_json

REGISTRY_CONTRACT="PIMS-ANOMALY-POC-CANDIDATE-REGISTRY-v1"


def build_registry(reviewed_path:Path,output_dir:Path,cluster_gap_hours:float,reference_phases:Path|None,overwrite:bool)->dict[str,Any]:
    require(cluster_gap_hours>0,"CANDIDATE_CLUSTER_GAP_INVALID",str(cluster_gap_hours)); output=ensure_output_directory(output_dir,overwrite); frame=pd.read_csv(reviewed_path,dtype=str).fillna(""); required=["candidate_uid","asset_id","episode_start","episode_end","field_decision","top_trigger_1","process_context","reviewer_reference","reviewed_at"]; require_columns(frame.columns,required,"REVIEWED_CANDIDATE_HEADER_INVALID")
    confirmed=frame[frame["field_decision"]=="CONFIRM_POC_CANDIDATE"].copy(); require(not confirmed.empty,"NO_FIELD_CONFIRMED_POC_CANDIDATES",str(reviewed_path)); confirmed["start_utc"]=pd.to_datetime(confirmed["episode_start"],utc=True); confirmed["end_utc"]=pd.to_datetime(confirmed["episode_end"],utc=True)
    cluster_values={}
    for asset,group in confirmed.groupby("asset_id",sort=True):
        cluster_number=0; previous_end=None
        for row in group.sort_values("start_utc").itertuples():
            if previous_end is None or row.start_utc-previous_end>pd.Timedelta(hours=cluster_gap_hours): cluster_number+=1
            cluster_values[row.candidate_uid]=f"POC-CLUSTER-{hashlib.sha256(f'{asset}|{cluster_number}|{row.episode_start}'.encode()).hexdigest()[:12]}"; previous_end=max(previous_end,row.end_utc) if previous_end is not None else row.end_utc
    confirmed["candidate_cluster_uid"]=confirmed["candidate_uid"].map(cluster_values); confirmed["reference_temporal_overlap"]="NONE"
    if reference_phases is not None:
        phases=pd.read_csv(reference_phases,dtype=str).fillna(""); require_columns(phases.columns,["event_uid","asset_id","start_time","end_time"],"PHASE_HEADER_INVALID"); phases["start_utc"]=pd.to_datetime(phases["start_time"],utc=True); phases["end_utc"]=pd.to_datetime(phases["end_time"],utc=True)
        overlaps=[]
        for row in confirmed.itertuples():
            matched=phases[(phases["asset_id"]==row.asset_id)&(phases["start_utc"]<row.end_utc)&(phases["end_utc"]>row.start_utc)]["event_uid"].unique(); overlaps.append("|".join(sorted(matched)) if len(matched) else "NONE")
        confirmed["reference_temporal_overlap"]=overlaps
    confirmed["registry_status"]="FIELD_CONFIRMED_POC_CANDIDATE"; confirmed["failure_confirmed"]=False; confirmed["causality_inferred"]=False; confirmed["production_use_allowed"]=False
    drop_columns=[column for column in ["start_utc","end_utc"] if column in confirmed]; registry=confirmed.drop(columns=drop_columns); registry_path=output/"field_confirmed_candidate_registry.csv"; registry.to_csv(registry_path,index=False)
    lines=[]
    for asset,group in registry.groupby("asset_id",sort=True): lines.append(f"VC19A|{asset}|CONF:{len(group)}|CLUSTERS:{group['candidate_cluster_uid'].nunique()}|REFERENCE_OVERLAP:{int((group['reference_temporal_overlap']!='NONE').sum())}")
    trigger_counts=registry.groupby(["asset_id","top_trigger_1"]).size().sort_values(ascending=False)
    for (asset,trigger),count in trigger_counts.items(): lines.append(f"VC19T|{asset}|TRIGGER:{trigger}|EP:{int(count)}")
    context_counts={}
    for value in registry["process_context"]:
        for token in str(value).split(";"):
            code=token.strip().split("=",1)[0]
            if code: context_counts[code]=context_counts.get(code,0)+1
    for code,count in sorted(context_counts.items(),key=lambda item:(-item[1],item[0])): lines.append(f"VC19C|CONTEXT:{code}|EP:{count}")
    payload={"contract_version":REGISTRY_CONTRACT,"reviewed_source_sha256":sha256_file(reviewed_path),"reference_phases_sha256":sha256_file(reference_phases) if reference_phases else None,"candidate_count":len(registry),"cluster_count":registry["candidate_cluster_uid"].nunique(),"asset_count":registry["asset_id"].nunique(),"cluster_gap_hours":cluster_gap_hours,"detail_lines":lines,"output_sha256":sha256_file(registry_path),"safety":{"failure_confirmed":False,"causality_inferred":False,"production_use_allowed":False,"external_write_allowed":False}}
    sha=canonical_sha256(payload); manifest={**payload,"receipt_sha256":sha}; write_json(output/"vc19_manifest.json",manifest); receipt=f"VC19=VALID|ASSETS:{payload['asset_count']}|CONF:{len(registry)}|CLUSTERS:{payload['cluster_count']}|SHA:{sha}"; (output/"vc19_receipt.txt").write_text(receipt+"\n",encoding="utf-8"); (output/"vc19_compact_detail.txt").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(receipt); print("\n".join(lines)); return manifest


def parser()->argparse.ArgumentParser:
    value=argparse.ArgumentParser(description=__doc__); value.add_argument("--reviewed",type=Path,required=True); value.add_argument("--output",type=Path,required=True); value.add_argument("--cluster-gap-hours",type=float,default=24); value.add_argument("--reference-phases",type=Path); value.add_argument("--overwrite",action="store_true"); return value


def main()->int:
    args=parser().parse_args(); build_registry(args.reviewed,args.output,args.cluster_gap_hours,args.reference_phases,args.overwrite); return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except AnomalyContractError as exc:
        print(f"CANDIDATE_REGISTRY=BLOCKED|CODE:{exc.code}"); raise SystemExit(2) from None
