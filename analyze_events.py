import json
import datetime
import argparse
import os
from collections import defaultdict

def parse_date(date_str: str) -> datetime.datetime:
    try:
        # LongMemEval format: "YYYY/MM/DD Day HH:MM" -> "YYYY/MM/DD HH:MM"
        parts = date_str.split(" ")
        clean_str = f"{parts[0]} {parts[2]}" 
        return datetime.datetime.strptime(clean_str, "%Y/%m/%d %H:%M")
    except Exception as e:
        return None

def analyze_dataset(data_path, max_users=50, max_sessions=10):
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    all_events = []
    
    # Process like the profiler does
    for item in data[:max_users]:
        uid = item.get("question_id")
        
        # History (Archives)
        sessions = item.get("haystack_sessions", [])[:max_sessions]
        dates = item.get("haystack_dates", [])
        for s, d in zip(sessions, dates):
            dt = parse_date(d)
            if dt:
                all_events.append({
                    "dt": dt,
                    "type": "archive",
                    "user_id": uid
                })
        
        # Queries
        q_date = item.get("question_date")
        if q_date:
            dt = parse_date(q_date)
            if dt:
                all_events.append({
                    "dt": dt,
                    "type": "query",
                    "user_id": uid
                })

    all_events.sort(key=lambda x: x["dt"])
    
    if not all_events:
        print("No events found.")
        return

    total_events = len(all_events)
    start_time = all_events[0]["dt"]
    end_time = all_events[-1]["dt"]
    duration_days = (end_time - start_time).days or 1

    print(f"--- Dataset Summary ---")
    print(f"Total Events: {total_events}")
    print(f"Unique Tenants: {len(set(e['user_id'] for e in all_events))}")
    print(f"Time Range: {start_time} to {end_time} ({duration_days} days)")
    print(f"Average Density: {total_events / duration_days:.2f} events/day")
    print("-" * 30)

    # Group by Day to find bursts
    daily_stats = defaultdict(lambda: {"count": 0, "users": set(), "types": defaultdict(int)})
    for e in all_events:
        day_str = e["dt"].strftime("%Y-%m-%d")
        daily_stats[day_str]["count"] += 1
        daily_stats[day_str]["users"].add(e["user_id"])
        daily_stats[day_str]["types"][e["type"]] += 1

    # Find "Best Areas"
    # Criteria: High density, multiple users, mix of types
    scored_days = []
    for day, stats in daily_stats.items():
        score = stats["count"] * len(stats["users"]) # Density * Diversity
        # Bonus for mix
        if stats["types"]["archive"] > 0 and stats["types"]["query"] > 0:
            score *= 1.5
        scored_days.append((day, stats, score))

    scored_days.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 Suggested Profiling Windows (Single Day):")
    for i, (day, stats, score) in enumerate(scored_days[:5]):
        print(f"{i+1}. Date: {day}")
        print(f"   Events: {stats['count']} (Archives: {stats['types']['archive']}, Queries: {stats['types']['query']})")
        print(f"   Unique Tenants: {len(stats['users'])}")
        print(f"   Recommended Command Flag: --start-date {day} --end-date {day}")
    
    # Also look for a sequence of days that gives ~100 events
    print("\nSuggested Range for ~100 Events:")
    target = 100
    best_range = None
    min_days = float('inf')
    
    all_days = sorted(daily_stats.keys())
    for i in range(len(all_days)):
        current_count = 0
        for j in range(i, len(all_days)):
            current_count += daily_stats[all_days[j]]["count"]
            if current_count >= target:
                num_days = j - i + 1
                if num_days < min_days:
                    min_days = num_days
                    best_range = (all_days[i], all_days[j], current_count)
                break
    
    if best_range:
        print(f"   Range: {best_range[0]} to {best_range[1]}")
        print(f"   Total Events: {best_range[2]}")
        print(f"   Recommended Command Flags: --start-date {best_range[0]} --end-date {best_range[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--max-users", type=int, default=50)
    parser.add_argument("--max-sessions", type=int, default=10)
    args = parser.parse_args()
    
    analyze_dataset(args.data_path, args.max_users, args.max_sessions)
