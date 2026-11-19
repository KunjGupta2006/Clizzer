from typing import List, Tuple


def plan_chunks(
    duration: float,
    window_sec: float = 1800.0,  # 30 minutes
    overlap_sec: float = 30.0,
    max_single_duration: float = 7200.0,  # 2 hours
) -> List[Tuple[float, float]]:
    if duration <= 0:
        return []

    # If duration <= 2 hours, no chunking needed
    if duration <= max_single_duration:
        return [(0.0, duration)]

    chunks: List[Tuple[float, float]] = []
    start = 0.0
    step = window_sec - overlap_sec

    while start < duration:
        end = min(start + window_sec, duration)
        chunks.append((start, end))
        if end >= duration:
            break
        start += step

    return chunks


def merge_chunk_timestamps(chunk_index: int, offset: float, timestamps: List[float]) -> List[float]:
    return [t + offset for t in timestamps]
