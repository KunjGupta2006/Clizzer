"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  getJob,
  getJobClips,
  getJobCurve,
  rerankJob,
  getWsUrl,
  JobData,
  ClipData,
  CurveData,
} from "@/lib/api";
import ProgressTimeline from "@/components/ProgressTimeline";
import ScoreCurve from "@/components/ScoreCurve";
import ClipCard from "@/components/ClipCard";
import { Sliders, RefreshCw, AlertCircle, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function JobDetailPage() {
  const params = useParams();
  const jobId = params?.id as string;

  const [job, setJob] = useState<JobData | null>(null);
  const [clips, setClips] = useState<ClipData[]>([]);
  const [curve, setCurve] = useState<CurveData | null>(null);
  const [loading, setLoading] = useState(true);

  // Rerank Controls
  const [rerankNum, setRerankNum] = useState(5);
  const [rerankMode, setRerankMode] = useState("auto");
  const [rerankLoading, setRerankLoading] = useState(false);
  const [weights, setWeights] = useState({
    audio: 0.25,
    chat: 0.25,
    text: 0.25,
    visual: 0.15,
    face: 0.10,
  });

  const fetchAllData = async () => {
    try {
      const jobData = await getJob(jobId);
      setJob(jobData);

      if (jobData.status === "done") {
        const [clipsData, curveData] = await Promise.all([
          getJobClips(jobId),
          getJobCurve(jobId),
        ]);
        setClips(clipsData);
        setCurve(curveData);
      }
    } catch (err) {
      console.error("Failed to load job data", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllData();

    // Setup WebSocket connection
    const wsUrl = getWsUrl(jobId);
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.stage || data.status) {
            setJob((prev) =>
              prev
                ? {
                    ...prev,
                    stage: data.stage || prev.stage,
                    progress: data.progress !== undefined ? data.progress : prev.progress,
                    status: data.status || prev.status,
                    timings: data.timings ? { ...prev.timings, ...data.timings } : prev.timings,
                    error: data.error !== undefined ? data.error : prev.error,
                  }
                : null
            );
            if (data.status === "done") {
              fetchAllData();
            }
          }
        } catch (e) {
          console.error("WS message parse error", e);
        }
      };
    } catch (e) {
      console.warn("WS connection failed, falling back to polling", e);
    }

    // Polling fallback every 3s until job finishes
    const interval = setInterval(() => {
      setJob((currentJob) => {
        if (!currentJob || (currentJob.status !== "done" && currentJob.status !== "failed")) {
          fetchAllData();
        }
        return currentJob;
      });
    }, 3000);

    return () => {
      clearInterval(interval);
      if (ws) ws.close();
    };
  }, [jobId]);

  const handleRerank = async () => {
    setRerankLoading(true);
    try {
      const updatedClips = await rerankJob(jobId, {
        num_clips: rerankNum,
        mode: rerankMode,
        weights,
      });
      setClips(updatedClips);
      const updatedCurve = await getJobCurve(jobId);
      setCurve(updatedCurve);
    } catch (err) {
      console.error("Reranking failed", err);
    } finally {
      setRerankLoading(false);
    }
  };

  if (loading && !job) {
    return (
      <div className="py-20 text-center text-gray-400 font-mono text-sm">
        Loading Job {jobId}...
      </div>
    );
  }

  if (!job) {
    return (
      <div className="max-w-md mx-auto text-center py-20">
        <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-white mb-2">Job Not Found</h2>
        <p className="text-sm text-gray-400 mb-6">The requested job ID could not be loaded.</p>
        <Link href="/" className="px-4 py-2 rounded-xl bg-gray-800 text-white font-medium text-xs">
          Return Home
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Back button */}
      <div>
        <Link
          href="/"
          className="inline-flex items-center space-x-1.5 text-xs text-gray-400 hover:text-white transition font-mono mb-2"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>New Clipping Job</span>
        </Link>
        <h1 className="text-2xl font-black text-white">
          Job <span className="font-mono text-pink-400">#{jobId}</span>
        </h1>
      </div>

      {/* Progress Timeline */}
      <ProgressTimeline
        status={job.status}
        stage={job.stage}
        progress={job.progress}
        timings={job.timings || {}}
        error={job.error}
      />

      {/* Multimodal Score Curve */}
      {curve && <ScoreCurve curve={curve} clips={clips} />}

      {/* Reranking Panel */}
      {job.status === "done" && (
        <div className="bg-[#14151d] border border-gray-800 rounded-2xl p-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-pink-500" />
              <span>Interactive Reranking Engine</span>
            </h2>
            <button
              onClick={handleRerank}
              disabled={rerankLoading}
              className="px-4 py-1.5 rounded-xl bg-pink-500 hover:bg-pink-600 disabled:opacity-50 text-white text-xs font-bold transition flex items-center space-x-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${rerankLoading ? "animate-spin" : ""}`} />
              <span>Apply Reranking</span>
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-[11px] font-mono text-gray-400 mb-1 uppercase">Clips Count</label>
              <input
                type="number"
                min={1}
                max={20}
                value={rerankNum}
                onChange={(e) => setRerankNum(parseInt(e.target.value) || 5)}
                className="w-full bg-[#0a0a0f] border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-white"
              />
            </div>

            <div>
              <label className="block text-[11px] font-mono text-gray-400 mb-1 uppercase">Preset Mode</label>
              <select
                value={rerankMode}
                onChange={(e) => setRerankMode(e.target.value)}
                className="w-full bg-[#0a0a0f] border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-white"
              >
                <option value="auto">Auto (Balanced)</option>
                <option value="hype">Hype</option>
                <option value="funny">Funny</option>
                <option value="informational">Informational</option>
              </select>
            </div>
          </div>

          {/* Weight Sliders */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2 border-t border-gray-800/80">
            {(["audio", "chat", "text", "visual", "face"] as const).map((m) => (
              <div key={m}>
                <div className="flex justify-between text-[10px] font-mono text-gray-400 mb-1 uppercase">
                  <span>{m}</span>
                  <span className="text-pink-400">{(weights[m] * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={weights[m]}
                  onChange={(e) =>
                    setWeights((prev) => ({ ...prev, [m]: parseFloat(e.target.value) }))
                  }
                  className="w-full accent-pink-500 cursor-pointer"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clips Grid */}
      {clips.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-extrabold text-white">
              Generated Short Clips ({clips.length})
            </h2>
            <span className="text-xs text-gray-500 font-mono">1080x1920 • -14 LUFS Loudnorm</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {clips.map((clip, idx) => (
              <ClipCard
                key={idx}
                clip={clip}
                index={idx}
                jobId={jobId}
                onClipUpdated={(updated) => {
                  setClips((prev) => {
                    const copy = [...prev];
                    copy[idx] = updated;
                    return copy;
                  });
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
