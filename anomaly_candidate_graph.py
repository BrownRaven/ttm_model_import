"""Create offline SVG evidence graphs and review sheets for M0 machine pre-candidates.

Graphs describe score-trigger evidence only. They never assert causality, equipment failure,
or production approval.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_contract import AnomalyContractError, canonical_sha256, ensure_output_directory, require, require_columns, sha256_file, write_json
from anomaly_validated_workflow import _bool

GRAPH_CONTRACT="PIMS-ANOMALY-CANDIDATE-GRAPH-v1"


def _episode_rows(frame:pd.DataFrame)->list[dict[str,Any]]:
    episodes=[]
    for asset,group in frame.groupby("asset_id",sort=True):
        group=group.copy(); group["start_utc"]=pd.to_datetime(group["window_start"],utc=True); group["end_utc"]=pd.to_datetime(group["window_end"],utc=True); group=group.sort_values("start_utc")
        active=[]; previous=None
        for row in group.itertuples(index=False):
            machine=_bool(row.machine_pre_candidate); contiguous=previous is not None and row.start_utc-previous<=pd.Timedelta(minutes=5)
            if machine and active and not contiguous: episodes.append(_summarize_episode(asset,active)); active=[]
            if machine: active.append(row)
            elif active: episodes.append(_summarize_episode(asset,active)); active=[]
            previous=row.start_utc
        if active: episodes.append(_summarize_episode(asset,active))
    return episodes


def _summarize_episode(asset:str,rows:list[Any])->dict[str,Any]:
    first,last=rows[0],rows[-1]; identity=f"{asset}|{first.window_start}|{last.window_end}"; uid="POC-CAND-"+hashlib.sha256(identity.encode()).hexdigest()[:12]
    trigger_counts:dict[tuple[str,str],int]={}
    for row in rows: trigger_counts[(str(row.trigger_sensor_uid),str(row.trigger_metric))]=trigger_counts.get((str(row.trigger_sensor_uid),str(row.trigger_metric)),0)+1
    triggers=sorted(trigger_counts.items(),key=lambda item:(-item[1],item[0]))[:3]; gate_statuses=sorted({str(row.state_gate_status) for row in rows}); decisions={str(row.decision) for row in rows}
    if "POC_ANOMALY_CANDIDATE" in decisions: review_status="MACHINE_AND_STATE_PASS"
    elif "POC_ANOMALY_PRE_CANDIDATE_REVIEW_REQUIRED" in decisions: review_status="FIELD_REVIEW_REQUIRED"
    else: review_status="CONFIRMED_STATE_REJECTED"
    return {"candidate_uid":uid,"asset_id":asset,"episode_start":str(first.window_start),"episode_end":str(last.window_end),"candidate_window_count":len(rows),"maximum_machine_persistence":max(int(row.machine_persistence_count) for row in rows),"minimum_feature_coverage_ratio":min(float(row.feature_coverage_ratio) for row in rows),"top_trigger_1":f"{triggers[0][0][0]}::{triggers[0][0][1]}" if triggers else "NONE","top_trigger_2":f"{triggers[1][0][0]}::{triggers[1][0][1]}" if len(triggers)>1 else "NONE","top_trigger_3":f"{triggers[2][0][0]}::{triggers[2][0][1]}" if len(triggers)>2 else "NONE","state_gate_statuses":"|".join(gate_statuses),"health_states":"|".join(sorted({str(row.health_state) for row in rows})),"review_status":review_status,"causality_status":"NOT_INFERRED","production_use_allowed":False}


def _box(x:int,y:int,w:int,h:int,title:str,lines:list[str],fill:str)->str:
    text=[f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#334155" stroke-width="1.5"/>',f'<text x="{x+14}" y="{y+24}" font-size="15" font-weight="700" fill="#0f172a">{html.escape(title)}</text>']
    for index,line in enumerate(lines): text.append(f'<text x="{x+14}" y="{y+48+index*19}" font-size="12" fill="#334155">{html.escape(line)}</text>')
    return "".join(text)


def _edge(x1:int,y1:int,x2:int,y2:int,label:str)->str:
    middle=(x1+x2)//2
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/><text x="{middle}" y="{y1-6}" text-anchor="middle" font-size="10" fill="#475569">{html.escape(label)}</text>'


def render_svg(row:dict[str,Any],path:Path)->None:
    triggers=[row[f"top_trigger_{index}"] for index in range(1,4) if row[f"top_trigger_{index}"]!="NONE"]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="700" viewBox="0 0 1180 700">','<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#64748b"/></marker></defs>','<rect width="1180" height="700" fill="#f8fafc"/>',f'<text x="40" y="40" font-size="22" font-weight="700" fill="#0f172a">PoC Anomaly Candidate Evidence Graph</text>',f'<text x="40" y="64" font-size="12" fill="#b91c1c">Score evidence only · NOT failure · NOT causality · NOT production action</text>']
    parts.append(_box(430,90,320,115,"Candidate episode",[row["candidate_uid"],f"status: {row['review_status']}",f"windows: {row['candidate_window_count']} / max persistence: {row['maximum_machine_persistence']}"],"#fee2e2"))
    parts.append(_box(40,260,300,120,"Observed asset/time",[row["asset_id"],f"start: {row['episode_start']}",f"end: {row['episode_end']}"],"#dbeafe"))
    parts.append(_box(440,260,300,140,"Machine evidence",[*triggers,f"minimum Feature coverage: {row['minimum_feature_coverage_ratio']:.3f}"],"#dcfce7"))
    parts.append(_box(840,260,300,140,"Field context",[f"state gate: {row['state_gate_statuses']}",f"health: {row['health_states']}","health is preserved, not inferred"],"#fef3c7"))
    parts.append(_box(340,500,500,145,"Required field review",["1. 실제 RUNNING 구간인가?","2. 정비·기동·정지·부하변화가 있었는가?","3. 부품/센서 epoch가 bundle과 같은가?","4. 센서/통신 이상인가, 공정 이탈 후보인가?","5. CONFIRM / REJECT / INCONCLUSIVE"],"#ede9fe"))
    parts.extend([_edge(590,205,190,260,"OBSERVED_ON"),_edge(590,205,590,260,"SUPPORTED_BY"),_edge(590,205,990,260,"HAS_CONTEXT"),_edge(590,400,590,500,"REQUIRES_REVIEW")])
    parts.append('</svg>'); path.write_text("".join(parts),encoding="utf-8")


def build_graphs(decisions_path:Path,output_dir:Path,overwrite:bool)->dict[str,Any]:
    output=ensure_output_directory(output_dir,overwrite); frame=pd.read_csv(decisions_path,low_memory=False); required=["asset_id","window_start","window_end","machine_pre_candidate","machine_persistence_count","threshold_exceeded","feature_coverage_ratio","feature_coverage_eligible","trigger_sensor_uid","trigger_metric","state_gate_status","health_state","decision"]; require_columns(frame.columns,required,"CANDIDATE_GRAPH_INPUT_INVALID")
    episodes=_episode_rows(frame); episode_columns=["candidate_uid","asset_id","episode_start","episode_end","candidate_window_count","maximum_machine_persistence","minimum_feature_coverage_ratio","top_trigger_1","top_trigger_2","top_trigger_3","state_gate_statuses","health_states","review_status","causality_status","production_use_allowed"]; episode_frame=pd.DataFrame(episodes,columns=episode_columns); episode_path=output/"candidate_episode_index.csv"; episode_frame.to_csv(episode_path,index=False)
    review=episode_frame[["candidate_uid","asset_id","episode_start","episode_end","review_status"]].copy(); review["field_decision"]="PENDING"; review["confirmed_operation_mode"]="UNKNOWN"; review["maintenance_overlap"]="UNKNOWN"; review["epoch_review_status"]="UNKNOWN"; review["sensor_condition"]="UNKNOWN"; review["process_context"]=""; review["reviewer_reference"]=""; review["reviewed_at"]=""; review["review_notes"]=""; review_path=output/"candidate_field_review_template.csv"; review.to_csv(review_path,index=False)
    graph_dir=output/"graphs"; graph_dir.mkdir();
    for row in episodes: render_svg(row,graph_dir/f"{row['candidate_uid']}.svg")
    counts=episode_frame["review_status"].value_counts(); lines=[]
    for asset in sorted(frame["asset_id"].astype(str).unique()):
        source=frame[frame["asset_id"].astype(str)==asset]; group=episode_frame[episode_frame["asset_id"].astype(str)==asset]
        exceed=int(source["threshold_exceeded"].map(_bool).sum()); feature_block=int((~source["feature_coverage_eligible"].map(_bool)).sum()); pre=int(source["machine_pre_candidate"].map(_bool).sum())
        if len(group): reason="HAS_MACHINE_PRE_CANDIDATE"
        elif exceed==0: reason="NO_THRESHOLD_EXCEEDANCE"
        elif feature_block>0: reason="PERSISTENCE_NOT_MET_OR_FEATURE_BLOCKED"
        else: reason="PERSISTENCE_NOT_MET"
        lines.append(f"VC16A|{asset}|W:{len(source)}|EXCEED:{exceed}|FEATURE_BLOCK:{feature_block}|PREW:{pre}|EP:{len(group)}|PASS:{int((group['review_status']=='MACHINE_AND_STATE_PASS').sum())}|REVIEW:{int((group['review_status']=='FIELD_REVIEW_REQUIRED').sum())}|STATE_REJECT:{int((group['review_status']=='CONFIRMED_STATE_REJECTED').sum())}|REASON:{reason}")
    payload={"contract_version":GRAPH_CONTRACT,"source_sha256":sha256_file(decisions_path),"episode_count":len(episodes),"graph_count":len(episodes),"review_required_count":int(counts.get("FIELD_REVIEW_REQUIRED",0)),"state_pass_count":int(counts.get("MACHINE_AND_STATE_PASS",0)),"state_rejected_count":int(counts.get("CONFIRMED_STATE_REJECTED",0)),"detail_lines":lines,"outputs":{"index_sha256":sha256_file(episode_path),"review_template_sha256":sha256_file(review_path)},"safety":{"causality_inferred":False,"failure_confirmed":False,"production_use_allowed":False,"external_write_allowed":False}}
    sha=canonical_sha256(payload); manifest={**payload,"receipt_sha256":sha}; write_json(output/"vc16_manifest.json",manifest); receipt=f"VC16=VALID|EP:{len(episodes)}|GRAPHS:{len(episodes)}|PASS:{payload['state_pass_count']}|REVIEW:{payload['review_required_count']}|STATE_REJECT:{payload['state_rejected_count']}|SHA:{sha}"; (output/"vc16_receipt.txt").write_text(receipt+"\n",encoding="utf-8"); (output/"vc16_compact_detail.txt").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(receipt); print("\n".join(lines)); return manifest


def parser()->argparse.ArgumentParser:
    value=argparse.ArgumentParser(description=__doc__); value.add_argument("--decisions",type=Path,required=True); value.add_argument("--output",type=Path,required=True); value.add_argument("--overwrite",action="store_true"); return value


def main()->int:
    args=parser().parse_args(); build_graphs(args.decisions,args.output,args.overwrite); return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except AnomalyContractError as exc:
        print(f"CANDIDATE_GRAPH=BLOCKED|CODE:{exc.code}"); raise SystemExit(2) from None
