"""Natural Language LLM Query Agent for Campus Guard event logs.

Allows querying output/events.jsonl using local Ollama (e.g. qwen3.5:4b)
with two-step intent-to-filter extraction and natural language response synthesis.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("query_agent")

DEFAULT_OLLAMA_URLS = [
    os.getenv("OLLAMA_URL"),
    "http://127.0.0.1:11434",
    "http://host.docker.internal:11434",
    "http://localhost:11434",
]


class EventQueryAgent:
    """Natural language interface over events.jsonl using local Ollama LLM."""

    def __init__(
        self,
        events_path: str | Path = "output/events.jsonl",
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:4b",
    ) -> None:
        self.events_path = Path(events_path)
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._active_url: Optional[str] = None

    def _resolve_ollama_url(self) -> Optional[str]:
        """Detect live Ollama endpoint among configured URL and candidate fallbacks."""
        if self._active_url is not None:
            return self._active_url

        candidates = [self.ollama_url] + [u for u in DEFAULT_OLLAMA_URLS if u and u != self.ollama_url]
        for url in candidates:
            try:
                req = urllib.request.Request(f"{url}/api/tags")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        self._active_url = url
                        return url
            except Exception:
                continue
        return None

    def _call_ollama(self, messages: list[dict], format_json: bool = False, temperature: float = 0.0, num_predict: int = 500) -> Optional[str]:
        """Call Ollama /api/chat with timeout and fallback."""
        url = self._resolve_ollama_url()
        if not url:
            return None

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if format_json:
            payload["format"] = "json"

        try:
            req = urllib.request.Request(
                f"{url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("message", {})
            content = msg.get("content", "").strip()
            thinking = msg.get("thinking", "").strip()

            if content:
                return content
            elif thinking:
                # If thinking model used tokens in thinking, extract JSON from thinking
                if format_json:
                    m = re.findall(r"\{.*?\}", thinking, re.DOTALL)
                    if m:
                        return m[-1]
                return thinking
            return None
        except Exception as exc:
            logger.debug(f"Ollama call failed on {url}: {exc}")
            return None

    def load_events(self) -> list[dict]:
        """Load all event records from events.jsonl."""
        if not self.events_path.exists():
            return []
        events: list[dict] = []
        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def extract_filters(self, user_query: str) -> dict:
        """Step 1: Extract structured query filters via LLM or keyword fallback."""
        system_prompt = (
            "You are a surveillance database filter extractor. Analyze the user query and convert it "
            "into a JSON filter object. Output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "camera_id": string or null,\n'
            '  "detection": "zone_intrusion" | "entry_exit" | "tool_slot" | "tamper" | null,\n'
            '  "event_type": string or null,\n'
            '  "verified_only": boolean or null,\n'
            '  "identity": string or null,\n'
            '  "min_video_time_s": number or null,\n'
            '  "max_video_time_s": number or null,\n'
            '  "sort_order": "latest" | "earliest" | "highest_confidence",\n'
            '  "limit": integer (default 10)\n'
            "}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        raw_res = self._call_ollama(messages, format_json=True, temperature=0.0, num_predict=300)
        if raw_res:
            try:
                # Clean code blocks
                clean = re.sub(r"^```(?:json)?", "", raw_res.strip(), flags=re.MULTILINE)
                clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE).strip()
                data = json.loads(clean)
                if isinstance(data, dict):
                    # Set defaults
                    data.setdefault("camera_id", None)
                    data.setdefault("detection", None)
                    data.setdefault("event_type", None)
                    data.setdefault("verified_only", None)
                    data.setdefault("identity", None)
                    data.setdefault("min_video_time_s", None)
                    data.setdefault("max_video_time_s", None)
                    data.setdefault("sort_order", "latest")
                    data.setdefault("limit", 10)
                    return data
            except Exception:
                pass

        return self._heuristic_extract_filters(user_query)

    def _heuristic_extract_filters(self, query: str) -> dict:
        """Resilient regex/keyword filter extraction fallback."""
        q = query.lower()

        # Camera ID
        cam_match = re.search(r"\b(cam\d+)\b", q)
        camera_id = cam_match.group(1) if cam_match else None

        # Detection module
        detection = None
        if any(w in q for w in ["zone", "intrusion", "intrusions", "trespass", "tresspass", "restricted"]):
            detection = "zone_intrusion"
        elif any(w in q for w in ["entry", "entries", "exit", "exits", "tripwire", "crossing", "crossings", "door"]):
            detection = "entry_exit"
        elif any(w in q for w in ["tool", "tools", "slot", "slots", "hammer", "wrench", "drill"]):
            detection = "tool_slot"
        elif any(w in q for w in ["tamper", "tampers", "blur", "occlusion", "covered", "moved", "offline"]):
            detection = "tamper"

        # Event type
        event_type = None
        if "after_hours" in q or "after hours" in q:
            event_type = "after_hours_presence"
        elif "zone_intrusion" in q:
            event_type = "zone_intrusion"
        elif ("entry" in q or "entries" in q) and "exit" not in q and "exits" not in q:
            event_type = "entry"
        elif ("exit" in q or "exits" in q) and "entry" not in q and "entries" not in q:
            event_type = "exit"
        elif "tool_removed" in q or ("removed" in q and "tool" in q):
            event_type = "tool_removed"
        elif "tool_returned" in q or ("returned" in q and "tool" in q):
            event_type = "tool_returned"
        elif "tamper_blur" in q or "blur" in q:
            event_type = "tamper_blur"
        elif "tamper_covered" in q or "covered" in q:
            event_type = "tamper_covered"
        elif "tamper_moved" in q or "moved" in q:
            event_type = "tamper_moved"

        # Verification filter
        verified_only = None
        if any(w in q for w in ["verified", "confirmed", "valid"]):
            if "unverified" not in q and "not verified" not in q:
                verified_only = True
        elif any(w in q for w in ["unverified", "rejected", "false alarm", "dropped"]):
            verified_only = False

        # Identity search
        identity = None
        known_names = ["jhithesh", "sanjaay", "aariz", "gautham"]
        for name in known_names:
            if name in q:
                identity = name.capitalize()
                break
        if not identity:
            if "authorized" in q and "unauthorized" not in q:
                identity = "AUTHORIZED"
            elif "unauthorized" in q:
                identity = "UNAUTHORIZED"

        # Time range bounds
        min_time = None
        max_time = None
        after_m = re.search(r"(?:after|from|since)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?", q)
        if after_m:
            min_time = float(after_m.group(1))
        before_m = re.search(r"(?:before|until|under)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?", q)
        if before_m:
            max_time = float(before_m.group(1))

        # Sort order
        sort_order = "latest"
        if any(w in q for w in ["highest confidence", "most confident", "confidence"]):
            sort_order = "highest_confidence"
        elif any(w in q for w in ["first", "earliest", "oldest"]):
            sort_order = "earliest"

        # Limit
        limit = 10
        lim_m = re.search(r"\b(?:top|limit|first)\s+(\d+)\b", q)
        if lim_m:
            limit = int(lim_m.group(1))

        return {
            "camera_id": camera_id,
            "detection": detection,
            "event_type": event_type,
            "verified_only": verified_only,
            "identity": identity,
            "min_video_time_s": min_time,
            "max_video_time_s": max_time,
            "sort_order": sort_order,
            "limit": limit,
        }

    def compute_aggregates(self, events: list[dict]) -> dict:
        """Compute summary statistics across event list."""
        stats = {
            "total_events": len(events),
            "vlm_verified_events": 0,
            "vlm_rejected_events": 0,
            "counts_by_detection": {},
            "counts_by_event_type": {},
            "counts_by_camera": {},
        }
        for e in events:
            # Verification status
            vlm = e.get("vlm_verification") or (e.get("details", {}) or {}).get("vlm_verification")
            if isinstance(vlm, dict):
                if vlm.get("verified") is True:
                    stats["vlm_verified_events"] += 1
                elif vlm.get("verified") is False:
                    stats["vlm_rejected_events"] += 1

            det = e.get("detection", "unknown")
            stats["counts_by_detection"][det] = stats["counts_by_detection"].get(det, 0) + 1

            evt_t = e.get("event_type", "unknown")
            stats["counts_by_event_type"][evt_t] = stats["counts_by_event_type"].get(evt_t, 0) + 1

            cam = e.get("camera_id", "unknown")
            stats["counts_by_camera"][cam] = stats["counts_by_camera"].get(cam, 0) + 1

        return stats

    def filter_events(self, events: list[dict], filters: dict) -> list[dict]:
        """Step 2: Deterministic in-memory filtering based on extracted filter criteria."""
        res: list[dict] = []
        for e in events:
            # Camera ID
            if filters.get("camera_id") and e.get("camera_id") != filters["camera_id"]:
                continue

            # Detection
            if filters.get("detection") and e.get("detection") != filters["detection"]:
                continue

            # Event type
            if filters.get("event_type") and e.get("event_type") != filters["event_type"]:
                continue

            # Video time
            vt = float(e.get("video_time_s", 0.0))
            if filters.get("min_video_time_s") is not None and vt < filters["min_video_time_s"]:
                continue
            if filters.get("max_video_time_s") is not None and vt > filters["max_video_time_s"]:
                continue

            # VLM Verified
            if filters.get("verified_only") is not None:
                vlm = e.get("vlm_verification") or (e.get("details", {}) or {}).get("vlm_verification")
                is_v = vlm.get("verified") if isinstance(vlm, dict) else None
                if filters["verified_only"] and is_v is not True:
                    continue
                if not filters["verified_only"] and is_v is not False:
                    continue

            # Identity / Person
            if filters.get("identity"):
                target_id = filters["identity"].lower()
                det_info = e.get("details", {}) or {}
                person_info = det_info.get("person") or det_info.get("face_recognition") or {}
                name = str(person_info.get("name", "")).lower()
                status = str(person_info.get("status", "")).lower()
                note = str(e.get("note", "")).lower()

                if target_id not in name and target_id not in status and target_id not in note:
                    continue

            res.append(e)

        # Sorting
        sort_order = filters.get("sort_order", "latest")
        if sort_order == "earliest":
            res.sort(key=lambda x: float(x.get("video_time_s", 0.0)))
        elif sort_order == "highest_confidence":
            res.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        else:  # latest
            res.sort(key=lambda x: float(x.get("video_time_s", 0.0)), reverse=True)

        return res

    def query(self, user_query: str) -> dict:
        """Execute full query: Intent-to-filter -> Filter -> Summarize via Ollama."""
        all_events = self.load_events()
        global_stats = self.compute_aggregates(all_events)
        filters = self.extract_filters(user_query)

        matched_events = self.filter_events(all_events, filters)
        limit = filters.get("limit", 10)
        display_subset = matched_events[:limit]

        # Compact format for LLM context
        compact_records = []
        for e in display_subset:
            vlm = e.get("vlm_verification") or (e.get("details", {}) or {}).get("vlm_verification") or {}
            person = (e.get("details", {}) or {}).get("person") or (e.get("details", {}) or {}).get("face_recognition") or {}
            compact_records.append({
                "event_id": e.get("event_id"),
                "camera_id": e.get("camera_id"),
                "detection": e.get("detection"),
                "event_type": e.get("event_type"),
                "video_time_s": e.get("video_time_s"),
                "confidence": e.get("confidence"),
                "vlm_verified": vlm.get("verified"),
                "vlm_reason": vlm.get("reason"),
                "person": person.get("name"),
                "snapshot_path": e.get("snapshot_path"),
            })

        # Step 3: Natural language synthesis via Ollama
        system_prompt = (
            "You are the Campus Guard AI Query Assistant. Synthesize a concise, clear answer "
            "to the user query based ONLY on the provided database summary and event records.\n"
            "Rules:\n"
            "- Directly answer the user query.\n"
            "- Explicitly cite timestamps (seconds), camera IDs, event types, and snapshot paths.\n"
            "- If 0 events match, clearly explain what was searched."
        )
        user_content = (
            f"User Query: {user_query}\n\n"
            f"Global Database Summary: {json.dumps(global_stats)}\n\n"
            f"Applied Filters: {json.dumps(filters)}\n\n"
            f"Matching Events ({len(matched_events)} total matches, showing {len(compact_records)}):\n"
            f"{json.dumps(compact_records, indent=2)}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        llm_answer = self._call_ollama(messages, format_json=False, temperature=0.1, num_predict=400)

        if not llm_answer:
            # Resilient fallback synthesis
            llm_answer = self._synthesize_fallback_answer(user_query, filters, global_stats, matched_events)

        snapshots = [
            e.get("snapshot_path") for e in display_subset if e.get("snapshot_path")
        ]

        return {
            "answer": llm_answer,
            "filters": filters,
            "global_stats": global_stats,
            "total_matched": len(matched_events),
            "matched_events": display_subset,
            "snapshots": snapshots,
        }

    def _synthesize_fallback_answer(
        self,
        query: str,
        filters: dict,
        stats: dict,
        matched: list[dict],
    ) -> str:
        """Rule-based natural language response when Ollama is offline."""
        lines = [f"### Campus Guard Query Summary for: *\"{query}\"*\n"]

        if not matched:
            applied = ", ".join(f"{k}={v}" for k, v in filters.items() if v is not None and k != "limit")
            lines.append(f"No events matched the criteria ({applied or 'all criteria'}).")
            lines.append(f"Global Database: {stats.get('total_events', 0)} total events logged.")
            return "\n".join(lines)

        lines.append(f"Found **{len(matched)}** matching event(s). Global database contains {stats.get('total_events', 0)} events.\n")
        lines.append("| Event ID | Camera | Detection | Event Type | Time (s) | Conf | VLM Verified | Snapshot |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for e in matched[:10]:
            vlm = e.get("vlm_verification") or (e.get("details", {}) or {}).get("vlm_verification") or {}
            v_str = "✅ Yes" if vlm.get("verified") else "❌ No" if vlm.get("verified") is False else "—"
            lines.append(
                f"| `{e.get('event_id')}` | `{e.get('camera_id')}` | `{e.get('detection')}` | "
                f"`{e.get('event_type')}` | {e.get('video_time_s', 0):.2f}s | "
                f"{e.get('confidence', 0):.2f} | {v_str} | `{e.get('snapshot_path')}` |"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query events.jsonl in natural language using local Ollama")
    parser.add_argument("query", type=str, help="Natural language query (e.g. 'Show all zone intrusions on cam01')")
    parser.add_argument("--events-file", default="output/events.jsonl", help="Path to events.jsonl")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument("--model", default="qwen3.5:4b", help="Model name")

    args = parser.parse_args()

    agent = EventQueryAgent(
        events_path=args.events_file,
        ollama_url=args.ollama_url,
        model=args.model,
    )
    result = agent.query(args.query)

    print("\n" + "=" * 80)
    print("                    CAMPUS GUARD NATURAL LANGUAGE QUERY")
    print("=" * 80)
    print(f"Query           : {args.query}")
    print(f"Extracted Filter: {json.dumps(result['filters'])}")
    print(f"Total Matched   : {result['total_matched']}")
    print("-" * 80)
    print(f"\nAnswer:\n{result['answer']}\n")
    if result["snapshots"]:
        print("Associated Snapshots:")
        for s in result["snapshots"][:5]:
            print(f"  - {s}")
    print("=" * 80)
