import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("VLM_MODEL", "qwen3.5:4b")

FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "camera_id": {"type": ["string", "null"]},
        "detection": {
            "type": ["string", "null"],
            "enum": ["zone_intrusion", "entry_exit", "tool_slot", "tamper", None],
        },
        "event_type": {"type": ["string", "null"]},
        "verified_only": {"type": ["boolean", "null"]},
        "identity": {"type": ["string", "null"]},
        "min_video_time_s": {"type": ["number", "null"]},
        "max_video_time_s": {"type": ["number", "null"]},
        "sort_order": {
            "type": "string",
            "enum": ["latest", "earliest", "highest_confidence"],
        },
        "limit": {"type": "integer"},
    },
    "required": [
        "camera_id",
        "detection",
        "event_type",
        "verified_only",
        "identity",
        "min_video_time_s",
        "max_video_time_s",
        "sort_order",
        "limit",
    ],
}


class EventQueryAgent:
    def __init__(
        self,
        events_path: str = "output/events.jsonl",
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.events_path = events_path
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    def load_events(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return []
        events = []
        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    @staticmethod
    def _clean_response(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "Thinking Process:" in text:
            for marker in ["Final Answer:", "Answer:", "Based on", "Here are", "There are"]:
                if marker in text:
                    return text[text.rfind(marker):].strip()
        return text

    def _call_ollama_chat(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": 1024},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return self._clean_response(data.get("message", {}).get("content", ""))

    def extract_filters(self, user_query: str) -> Dict[str, Any]:
        q_lower = user_query.lower()
        system_prompt = (
            "/no_think\n"
            "You are a security database query parser for Campus Guard.\n"
            "Convert the user's question into a JSON filter object.\n"
            "- Valid detections: 'zone_intrusion', 'entry_exit', 'tool_slot', 'tamper'.\n"
            "- IMPORTANT: Keep 'event_type' as null if 'detection' already covers the request (e.g. for zone_intrusion logs, set detection='zone_intrusion' and event_type=null).\n"
            "- IMPORTANT: Set 'verified_only' to null UNLESS the user explicitly says 'verified', 'confirmed', or 'rejected'.\n"
            "- Set all other unused filter fields to null. Default limit is 15, default sort_order is 'latest'."
        )
        try:
            raw = self._call_ollama_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                json_schema=FILTER_SCHEMA,
                temperature=0.0,
            )
            filters = json.loads(raw)
        except Exception:
            filters = {
                "camera_id": "cam01" if "cam01" in q_lower else None,
                "detection": None,
                "event_type": None,
                "verified_only": None,
                "identity": None,
                "min_video_time_s": None,
                "max_video_time_s": None,
                "sort_order": "latest",
                "limit": 15,
            }

        # Guardrail 1: Never allow verified_only=True/False unless explicitly requested in query
        if not any(w in q_lower for w in ["verified", "confirmed", "rejected", "unverified", "false alarm"]):
            filters["verified_only"] = None

        # Guardrail 2: Ensure obvious detection keywords always map cleanly
        if not filters.get("detection"):
            if "zone_intrusion" in q_lower or "intrusion" in q_lower or "restricted zone" in q_lower:
                filters["detection"] = "zone_intrusion"
            elif "entry_exit" in q_lower or ("entry" in q_lower and "exit" in q_lower):
                filters["detection"] = "entry_exit"
            elif "tool_slot" in q_lower or "tool" in q_lower:
                filters["detection"] = "tool_slot"
            elif "tamper" in q_lower:
                filters["detection"] = "tamper"

        # Guardrail 3: Avoid redundant event_type mismatch when querying a whole detection category
        if filters.get("detection") == "zone_intrusion" and "after_hours" not in q_lower and "after hours" not in q_lower:
            filters["event_type"] = None

        if not filters.get("limit") or int(filters["limit"]) < 1:
            filters["limit"] = 15

        return filters

    @staticmethod
    def _get_vlm_status(evt: Dict[str, Any]) -> Optional[bool]:
        details = evt.get("details") or {}
        vlm = details.get("vlm_verification") or evt.get("vlm_verification")
        if isinstance(vlm, dict) and "verified" in vlm:
            return bool(vlm["verified"])
        return None

    @staticmethod
    def _get_snapshot(evt: Dict[str, Any]) -> Optional[str]:
        details = evt.get("details") or {}
        vlm = details.get("vlm_verification") or evt.get("vlm_verification") or {}
        snap = evt.get("snapshot_path") or vlm.get("snapshot_path")
        return snap if snap and snap != "none" else None

    def filter_events(
        self, events: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        matched = []
        for evt in events:
            if filters.get("camera_id") and evt.get("camera_id") != filters["camera_id"]:
                continue
            if filters.get("detection") and evt.get("detection") != filters["detection"]:
                continue
            if filters.get("event_type") and evt.get("event_type") != filters["event_type"]:
                continue
            if filters.get("verified_only") is not None:
                vlm_status = self._get_vlm_status(evt)
                # If event has VLM status, enforce match; if VLM wasn't run on it, treat as unverified only when looking for rejected=False
                effective_status = True if vlm_status is None else vlm_status
                if effective_status != filters["verified_only"]:
                    continue
            if filters.get("min_video_time_s") is not None:
                if float(evt.get("video_time_s", 0.0)) < float(filters["min_video_time_s"]):
                    continue
            if filters.get("max_video_time_s") is not None:
                if float(evt.get("video_time_s", 0.0)) > float(filters["max_video_time_s"]):
                    continue
            if filters.get("identity"):
                face_info = (evt.get("details") or {}).get("face_recognition") or {}
                if filters["identity"].lower() not in json.dumps(face_info).lower():
                    continue
            matched.append(evt)

        sort_order = filters.get("sort_order", "latest")
        if sort_order == "highest_confidence":
            matched.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        elif sort_order == "earliest":
            matched.sort(key=lambda x: float(x.get("video_time_s", 0.0)))
        else:
            matched.sort(key=lambda x: float(x.get("video_time_s", 0.0)), reverse=True)

        return matched

    def compute_aggregates(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_detection: Dict[str, int] = {}
        by_event_type: Dict[str, int] = {}
        by_camera: Dict[str, int] = {}
        verified_count = 0
        rejected_count = 0

        for e in events:
            det = e.get("detection", "unknown")
            et = e.get("event_type", "unknown")
            cam = e.get("camera_id", "unknown")
            by_detection[det] = by_detection.get(det, 0) + 1
            by_event_type[et] = by_event_type.get(et, 0) + 1
            by_camera[cam] = by_camera.get(cam, 0) + 1
            st = self._get_vlm_status(e)
            if st is True:
                verified_count += 1
            elif st is False:
                rejected_count += 1

        return {
            "total_events": len(events),
            "vlm_verified_events": verified_count,
            "vlm_rejected_events": rejected_count,
            "counts_by_detection": by_detection,
            "counts_by_event_type": by_event_type,
            "counts_by_camera": by_camera,
        }

    def query(self, user_query: str) -> Dict[str, Any]:
        all_events = self.load_events()
        if not all_events:
            return {
                "answer": f"No events found in `{self.events_path}`.",
                "filters": {},
                "global_stats": {},
                "total_matched": 0,
                "matched_events": [],
                "snapshots": [],
            }

        filters = self.extract_filters(user_query)
        matched = self.filter_events(all_events, filters)
        limit = max(1, min(int(filters.get("limit") or 15), 30))
        top_matches = matched[:limit]
        global_stats = self.compute_aggregates(all_events)

        compact_records = []
        snapshots = []
        unique_tracks = set()
        for e in top_matches:
            details = e.get("details") or {}
            vlm = details.get("vlm_verification") or e.get("vlm_verification") or {}
            face = details.get("face_recognition") or {}
            track_id = details.get("track_id")
            if track_id is not None:
                unique_tracks.add(track_id)
            snap = self._get_snapshot(e)
            if snap:
                snapshots.append(snap)
            compact_records.append({
                "event_id": e.get("event_id"),
                "camera_id": e.get("camera_id"),
                "detection": e.get("detection"),
                "event_type": e.get("event_type"),
                "video_time_s": e.get("video_time_s"),
                "confidence": e.get("confidence"),
                "track_id": track_id,
                "zone_id": details.get("zone_id"),
                "slot_id": details.get("slot_id"),
                "vlm_verified": vlm.get("verified"),
                "vlm_reason": vlm.get("reason") or e.get("note"),
                "person": face.get("identity"),
                "snapshot_path": snap,
            })

        synth_system = (
            "/no_think\n"
            "You are the Campus Guard Security AI Assistant. Answer the user's question directly and concisely "
            "using ONLY the provided Global Database Summary and Matching Event Records. "
            "List the matched events with their timestamps (video_time_s), camera_id, track_id/zone_id, confidence, and snapshot_path if available."
        )
        synth_user = (
            f"User Question: {user_query}\n\n"
            f"Global Database Summary: {json.dumps(global_stats)}\n"
            f"Applied Filters: {json.dumps(filters)}\n"
            f"Total Matching Events: {len(matched)} (Unique track_ids in matches: {sorted(list(unique_tracks))})\n"
            f"Matching Event Records (showing {len(compact_records)}):\n"
            f"{json.dumps(compact_records, indent=2)}"
        )

        try:
            answer = self._call_ollama_chat(
                messages=[
                    {"role": "system", "content": synth_system},
                    {"role": "user", "content": synth_user},
                ],
                json_schema=None,
                temperature=0.1,
            )
        except Exception as err:
            answer = f"Matched {len(matched)} event(s). (Ollama synthesis error: {err})"

        return {
            "answer": answer,
            "filters": filters,
            "global_stats": global_stats,
            "total_matched": len(matched),
            "matched_events": top_matches,
            "snapshots": snapshots,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Campus Guard Natural Language Query Agent")
    parser.add_argument("question", type=str, help="Natural language question about events.jsonl")
    parser.add_argument("--events", type=str, default="output/events.jsonl", help="Path to events.jsonl")
    args = parser.parse_args()

    agent = EventQueryAgent(events_path=args.events)
    res = agent.query(args.question)

    print("\n" + "=" * 80)
    print("                    CAMPUS GUARD NATURAL LANGUAGE QUERY")
    print("=" * 80)
    print(f"Query           : {args.question}")
    print(f"Extracted Filter: {json.dumps(res['filters'])}")
    print(f"Total Matched   : {res['total_matched']}")
    print("-" * 80)
    print(f"\nAnswer:\n{res['answer']}\n")
    if res["snapshots"]:
        print("-" * 80)
        print("Snapshots       :", ", ".join(res["snapshots"][:5]))
    print("=" * 80)
