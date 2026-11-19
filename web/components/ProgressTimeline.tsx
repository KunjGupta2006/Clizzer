"use client";

import React from "react";
import { CheckCircle2, Clock, AlertTriangle, Loader2 } from "lucide-react";

interface ProgressTimelineProps {
  status: string;
  stage: string;
  progress: number;
  timings: Record<string, number>;
  error?: string | null;
}

const STAGES = [
  { id: "queued", label: "Queued" },
  { id: "downloading", label: "Ingest Media & Chat" },
  { id: "transcribing", label: "ASR & Diarization" },
  { id: "analyzing", label: "Vision & Acoustics" },
  { id: "ranking", label: "Multimodal Fusion" },
  { id: "rendering", label: "Reframing & Captions" },
  { id: "done", label: "Clips Ready" },
];

export default function ProgressTimeline({
  status,
  stage,
  progress,
  timings,
  error,
}: ProgressTimelineProps) {
  const currentStageIndex = STAGES.findIndex((s) => s.id === stage);

  return (
    <div className="bg-[#14151d] border border-gray-800 rounded-2xl p-6 shadow-xl mb-8">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <span>Processing Pipeline</span>
            {status !== "done" && status !== "failed" && (
              <Loader2 className="w-4 h-4 text-pink-500 animate-spin" />
            )}
            {status === "failed" && (
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-red-500/20 text-red-400 font-mono font-bold uppercase">
                Failed
              </span>
            )}
          </h2>
          <p className="text-xs text-gray-400 font-mono mt-0.5">
            Stage:{" "}
            <span
              className={`uppercase font-bold ${
                status === "failed" ? "text-red-400" : "text-pink-400"
              }`}
            >
              {status === "failed" ? "Error / Interrupted" : stage}
            </span>{" "}
            ({Math.round(progress * 100)}%)
          </p>
        </div>

        {timings.total && (
          <div className="text-right text-xs font-mono text-gray-500">
            Total Time: <span className="text-gray-300 font-bold">{timings.total.toFixed(1)}s</span>
          </div>
        )}
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-[#0a0a0f] rounded-full h-2.5 mb-6 overflow-hidden border border-gray-800">
        <div
          className="bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 h-2.5 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${Math.min(100, Math.max(5, progress * 100))}%` }}
        />
      </div>

      {error && (
        <div className="mb-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex flex-col space-y-1.5">
          <div className="flex items-center space-x-2 font-semibold">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            <span>Pipeline error: {error}</span>
          </div>
          {error.includes("timed out") && (
            <p className="text-[11px] text-gray-300 ml-6">
              💡 <strong>Tip for long videos:</strong> The submitted video was very long (e.g. multi-hour stream). For fastest clipping on CPU, try videos under 1–2 hours or local video clips.
            </p>
          )}
        </div>
      )}

      {/* Steps List */}
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2">
        {STAGES.map((s, idx) => {
          const isDone = currentStageIndex > idx || status === "done";
          const isCurrent = currentStageIndex === idx && status !== "done";

          return (
            <div
              key={s.id}
              className={`p-2.5 rounded-xl border text-center transition ${
                isDone
                  ? "bg-green-500/5 border-green-500/20 text-green-400"
                  : isCurrent
                  ? "bg-pink-500/10 border-pink-500/30 text-pink-400 ring-1 ring-pink-500/20"
                  : "bg-[#0e0f15] border-gray-800 text-gray-500"
              }`}
            >
              <div className="text-[10px] font-mono uppercase mb-1">Step {idx + 1}</div>
              <div className="text-xs font-semibold truncate">{s.label}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
