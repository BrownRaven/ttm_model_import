"""Prepare and validate field review of VC20 trigger diagnostic figures."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_contract import AnomalyContractError, canonical_sha256, ensure_output_directory, require, require_columns, sha256_file, write_json

CONTRACT="PIMS-ANOMALY-TRIGGER-FIELD-REVIEW-v1"
ASSESSMENTS={"PENDING","VALID_CURRENT_TRANSIENT","SINGLE_SAMPLE_SPIKE_SUSPECTED","SENSOR_ARTIFACT_SUSPECTED","OPERATION_TRANSITION_SUSPECTED","RAW_1S_ZOOM_REQUIRED","INCONCLUSIVE"}
CROSS={"PRESENT","ABSENT","UNKNOWN"}; MEAN={"PRESENT","ABSENT","UNKNOWN"}; REPEAT={"REPEATED","ISOLATED","UNKNOWN"}; YES_NO={"YES","NO"}
FIELDS=["candidate_uid","asset_id","top_trigger","figure_file","trigger_assessment","cross_sensor_response","mean_shift_status","repeated_pattern_status","raw_1s_zoom_required","reviewer_reference","reviewed_at","review_notes"]


def prepare(index_path:Path,output_path:Path)->None:
    index=pd.read_csv(index_path,dtype=str).fillna(""); require_columns(index.columns,["candidate_uid","asset_id","top_trigger","figure_file"],"TRIGGER_FIGURE_INDEX_INVALID"); require(index["candidate_uid"].is_unique,"TRIGGER_INDEX_UID_DUPLICATE",str(index_path)); review=index[["candidate_uid","asset_id","top_trigger","figure_file"]].copy(); review["trigger_assessment"]="PENDING"; review["cross_sensor_response"]="UNKNOWN"; review["mean_shift_status"]="UNKNOWN"; review["repeated_pattern_status"]="UNKNOWN"; review["raw_1s_zoom_required"]="NO"; review["reviewer_reference"]=""; review["reviewed_at"]=""; review["review_notes"]=""; destination=output_path.expanduser().resolve(); destination.parent.mkdir(parents=True,exist_ok=True); review.to_csv(destination,index=False); print(f"VC21_TEMPLATE=VALID|ROWS:{len(review)}")


def validate(index_path:Path,reviews_path:Path,output_dir:Path,overwrite:bool)->dict[str,Any]:
    output=ensure_output_directory(output_dir,overwrite); index=pd.read_csv(index_path,dtype=str).fillna(""); reviews=pd.read_csv(reviews_path,dtype=str).fillna(""); require_columns(reviews.columns,FIELDS,"TRIGGER_REVIEW_HEADER_INVALID"); require(set(index["candidate_uid"])==set(reviews["candidate_uid"]),"TRIGGER_REVIEW_UID_SET_MISMATCH",str(reviews_path)); require(reviews["candidate_uid"].is_unique,"TRIGGER_REVIEW_UID_DUPLICATE",str(reviews_path)); issues=[]
    for row in reviews.to_dict(orient="records"):
        uid=row["candidate_uid"]; assessment=row["trigger_assessment"]
        if assessment not in ASSESSMENTS: issues.append((uid,"TRIGGER_ASSESSMENT_INVALID"))
        if row["cross_sensor_response"] not in CROSS: issues.append((uid,"CROSS_SENSOR_RESPONSE_INVALID"))
        if row["mean_shift_status"] not in MEAN: issues.append((uid,"MEAN_SHIFT_STATUS_INVALID"))
        if row["repeated_pattern_status"] not in REPEAT: issues.append((uid,"REPEATED_PATTERN_STATUS_INVALID"))
        if row["raw_1s_zoom_required"] not in YES_NO: issues.append((uid,"RAW_ZOOM_FLAG_INVALID"))
        if assessment=="PENDING": continue
        if not row["reviewer_reference"]: issues.append((uid,"REVIEWER_REFERENCE_REQUIRED"))
        timestamp=pd.to_datetime(row["reviewed_at"],errors="coerce");
        if not row["reviewed_at"] or pd.isna(timestamp) or timestamp.tzinfo is None: issues.append((uid,"REVIEWED_AT_OFFSET_REQUIRED"))
        if assessment in {"SINGLE_SAMPLE_SPIKE_SUSPECTED","SENSOR_ARTIFACT_SUSPECTED","RAW_1S_ZOOM_REQUIRED"} and row["raw_1s_zoom_required"]!="YES": issues.append((uid,"RAW_1S_ZOOM_REQUIRED_FOR_ASSESSMENT"))
    issue_frame=pd.DataFrame(issues,columns=["candidate_uid","issue_code"]); issue_path=output/"trigger_review_issues.csv"; issue_frame.to_csv(issue_path,index=False); require(issue_frame.empty,"TRIGGER_FIELD_REVIEW_INVALID",str(issue_path))
    result=reviews.copy(); result["failure_confirmed"]=False; result["causality_inferred"]=False; result["production_use_allowed"]=False; result_path=output/"reviewed_trigger_diagnostics.csv"; result.to_csv(result_path,index=False); lines=[]
    for asset,group in result.groupby("asset_id",sort=True):
        counts=group["trigger_assessment"].value_counts(); lines.append(f"VC21A|{asset}|TOTAL:{len(group)}|VALID:{int(counts.get('VALID_CURRENT_TRANSIENT',0))}|SPIKE:{int(counts.get('SINGLE_SAMPLE_SPIKE_SUSPECTED',0))}|ARTIFACT:{int(counts.get('SENSOR_ARTIFACT_SUSPECTED',0))}|TRANSITION:{int(counts.get('OPERATION_TRANSITION_SUSPECTED',0))}|ZOOM_ONLY:{int(counts.get('RAW_1S_ZOOM_REQUIRED',0))}|INCONCLUSIVE:{int(counts.get('INCONCLUSIVE',0))}|PENDING:{int(counts.get('PENDING',0))}|RAW_ZOOM:{int((group['raw_1s_zoom_required']=='YES').sum())}")
    counts=result["trigger_assessment"].value_counts(); payload={"contract_version":CONTRACT,"index_sha256":sha256_file(index_path),"review_sha256":sha256_file(reviews_path),"candidate_count":len(result),"pending_count":int(counts.get("PENDING",0)),"raw_zoom_count":int((result["raw_1s_zoom_required"]=="YES").sum()),"detail_lines":lines,"output_sha256":sha256_file(result_path),"safety":{"failure_confirmed":False,"causality_inferred":False,"production_use_allowed":False,"actual_values_included":False}}
    sha=canonical_sha256(payload); manifest={**payload,"receipt_sha256":sha}; write_json(output/"vc21_manifest.json",manifest); receipt=f"VC21=VALID|TOTAL:{len(result)}|PENDING:{payload['pending_count']}|RAW_ZOOM:{payload['raw_zoom_count']}|SHA:{sha}"; (output/"vc21_receipt.txt").write_text(receipt+"\n",encoding="utf-8"); (output/"vc21_compact_detail.txt").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(receipt); print("\n".join(lines)); return manifest


def parser()->argparse.ArgumentParser:
    value=argparse.ArgumentParser(description=__doc__); commands=value.add_subparsers(dest="command",required=True); first=commands.add_parser("prepare"); first.add_argument("--index",type=Path,required=True); first.add_argument("--output-file",type=Path,required=True); second=commands.add_parser("validate"); second.add_argument("--index",type=Path,required=True); second.add_argument("--reviews",type=Path,required=True); second.add_argument("--output",type=Path,required=True); second.add_argument("--overwrite",action="store_true"); return value


def main()->int:
    args=parser().parse_args(); prepare(args.index,args.output_file) if args.command=="prepare" else validate(args.index,args.reviews,args.output,args.overwrite); return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except AnomalyContractError as exc:
        print(f"TRIGGER_REVIEW=BLOCKED|CODE:{exc.code}"); raise SystemExit(2) from None
