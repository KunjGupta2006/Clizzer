"use client";

import React, { useState } from "react";
import { trimClip, ClipData } from "@/lib/api";
import { Scissors, Check, Loader2 } from "lucide-react";

interface TrimHandlesProps {
  clip: ClipData;
  clipId: string;
  onTrimmed: (updatedClip: ClipData) => void;
}

export default function TrimHandles({ clip, clipId, onTrimmed }: TrimHandlesProps) {
  const [startDelta, setStartDelta] = useState(0);
  const [endDelta, setEndDelta] = useState(0);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleApplyTrim = async () => {
    if (startDelta === 0 && endDelta === 0) return;
    setLoading(true);
    setSuccess(false);
    try {
      const updated = await trimClip(clipId, startDelta, endDelta);
      onTrimmed(updated);
      setStartDelta(0);
      setEndDelta(0);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 2000);
    } catch (err) {
      console.error("Trim failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4 p-3 rounded-xl bg-[#0a0a0f] border border-gray-800 text-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-gray-400 flex items-center space-x-1">
          <Scissors className="w-3.5 h-3.5 text-pink-500" />
          <span>Fine Trim (+/- 5s)</span>
        </span>
        <span className="font-mono text-gray-500">
          Duration: {(clip.end - clip.start + endDelta - startDelta).toFixed(1)}s
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <div>
          <div className="flex justify-between text-[11px] text-gray-400 mb-1">
            <span>Start Offset</span>
            <span className="font-mono text-pink-400">{startDelta > 0 ? `+${startDelta}s` : `${startDelta}s`}</span>
          </div>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.5}
            value={startDelta}
            onChange={(e) => setStartDelta(parseFloat(e.target.value))}
            className="w-full accent-pink-500 cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-[11px] text-gray-400 mb-1">
            <span>End Offset</span>
            <span className="font-mono text-cyan-400">{endDelta > 0 ? `+${endDelta}s` : `${endDelta}s`}</span>
          </div>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.5}
            value={endDelta}
            onChange={(e) => setEndDelta(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer"
          />
        </div>
      </div>

      <button
        onClick={handleApplyTrim}
        disabled={loading || (startDelta === 0 && endDelta === 0)}
        className="w-full py-1.5 px-3 rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-gray-200 font-semibold transition flex items-center justify-center space-x-1.5"
      >
        {loading ? (
          <>
            <Loader2 className="w-3 h-3 animate-spin text-pink-500" />
            <span>Re-encoding Clip...</span>
          </>
        ) : success ? (
          <>
            <Check className="w-3 h-3 text-green-400" />
            <span>Trim Saved!</span>
          </>
        ) : (
          <span>Apply Boundary Adjustment</span>
        )}
      </button>
    </div>
  );
}
