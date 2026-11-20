"use client";

import React, { useState } from "react";
import { ClipData, getClipDownloadUrl } from "@/lib/api";
import TrimHandles from "@/components/TrimHandles";
import { Copy, Check, Download, Play, Music, Award } from "lucide-react";

interface ClipCardProps {
  clip: ClipData;
  index: number;
  jobId: string;
  onClipUpdated: (updatedClip: ClipData) => void;
}

export default function ClipCard({ clip, index, jobId, onClipUpdated }: ClipCardProps) {
  const [copiedTitle, setCopiedTitle] = useState(false);
  const [copiedTags, setCopiedTags] = useState(false);

  const clipNumStr = String(index + 1).padStart(2, "0");
  const clipFilename = `clip_${clipNumStr}`;
  const clipId = `${jobId}_clip_${clipNumStr}`;
  const downloadUrl = getClipDownloadUrl(clipFilename, jobId);

  const copyText = (text: string, setter: (val: boolean) => void) => {
    navigator.clipboard.writeText(text);
    setter(true);
    setTimeout(() => setter(false), 2000);
  };

  const scoreBars = [
    { label: "Audio", val: clip.breakdown.audio || 0, color: "bg-purple-500" },
    { label: "Chat", val: clip.breakdown.chat || 0, color: "bg-cyan-400" },
    { label: "Text", val: clip.breakdown.text || 0, color: "bg-pink-500" },
    { label: "Visual", val: clip.breakdown.visual || 0, color: "bg-amber-400" },
    { label: "Face", val: clip.breakdown.face || 0, color: "bg-emerald-400" },
  ];

  return (
    <div className="bg-[#14151d] border border-gray-800 rounded-2xl overflow-hidden shadow-xl flex flex-col justify-between">
      {/* Video Preview - True 9:16 Vertical Shorts Aspect */}
      <div className="relative aspect-[9/16] w-full bg-black flex items-center justify-center overflow-hidden group rounded-t-2xl">
        <video
          src={downloadUrl}
          poster={clip.thumb ? `/api/proxy?path=${encodeURIComponent(clip.thumb)}` : undefined}
          controls
          playsInline
          className="w-full h-full object-contain"
        />

        {/* Score Badge */}
        <div className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-black/70 backdrop-blur border border-pink-500/40 text-pink-400 font-mono text-xs font-bold flex items-center space-x-1">
          <Award className="w-3.5 h-3.5" />
          <span>Score: {clip.fused_score.toFixed(2)}</span>
        </div>

        {clip.music_ratio > 0.6 && (
          <div className="absolute top-3 right-3 px-2 py-1 rounded-full bg-amber-500/20 backdrop-blur border border-amber-500/40 text-amber-300 font-mono text-[10px] flex items-center space-x-1">
            <Music className="w-3 h-3" />
            <span>Music &gt; 60%</span>
          </div>
        )}
      </div>

      {/* Details & Metadata */}
      <div className="p-5 flex-1 flex flex-col justify-between">
        <div className="space-y-3">
          {/* Title */}
          <div>
            <div className="flex items-center justify-between text-xs text-gray-500 font-mono mb-1">
              <span>Title</span>
              <button
                onClick={() => copyText(clip.title, setCopiedTitle)}
                className="text-gray-400 hover:text-white flex items-center space-x-1 transition"
              >
                {copiedTitle ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                <span>{copiedTitle ? "Copied" : "Copy"}</span>
              </button>
            </div>
            <h3 className="text-sm font-bold text-white line-clamp-2">{clip.title}</h3>
          </div>

          {/* Hook */}
          {clip.hook && (
            <div className="p-2.5 rounded-xl bg-[#0a0a0f] border border-gray-800/80">
              <span className="text-[10px] font-mono uppercase text-pink-400 block mb-0.5">2-Sec Hook</span>
              <p className="text-xs text-gray-300 italic">&ldquo;{clip.hook}&rdquo;</p>
            </div>
          )}

          {/* Hashtags */}
          {clip.hashtags && clip.hashtags.length > 0 && (
            <div>
              <div className="flex items-center justify-between text-xs text-gray-500 font-mono mb-1">
                <span>Hashtags</span>
                <button
                  onClick={() => copyText(clip.hashtags.join(" "), setCopiedTags)}
                  className="text-gray-400 hover:text-white flex items-center space-x-1 transition"
                >
                  {copiedTags ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedTags ? "Copied" : "Copy"}</span>
                </button>
              </div>
              <div className="flex flex-wrap gap-1">
                {clip.hashtags.map((tag, tIdx) => (
                  <span
                    key={tIdx}
                    className="text-[11px] px-2 py-0.5 rounded-md bg-gray-800/60 text-cyan-400 font-mono"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Modality Breakdown Bars */}
          <div className="pt-2">
            <span className="text-[10px] font-mono text-gray-500 uppercase block mb-1.5">
              Signal Breakdown
            </span>
            <div className="grid grid-cols-5 gap-1.5 text-center">
              {scoreBars.map((b) => {
                const normVal = Math.max(0.1, Math.min(1.0, (b.val + 3.0) / 6.0));
                return (
                  <div key={b.label}>
                    <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden mb-1">
                      <div
                        className={`h-full ${b.color}`}
                        style={{ width: `${Math.round(normVal * 100)}%` }}
                      />
                    </div>
                    <span className="text-[9px] text-gray-400 font-mono">{b.label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Trim & Download */}
        <div className="mt-4 pt-3 border-t border-gray-800/80">
          <TrimHandles
            clip={clip}
            clipId={`job_${jobId}_clip_${index + 1}`}
            onTrimmed={onClipUpdated}
          />

          <a
            href={downloadUrl}
            download={`clip_${index + 1}.mp4`}
            className="mt-3 w-full py-2 px-4 rounded-xl bg-pink-500/10 hover:bg-pink-500/20 text-pink-400 border border-pink-500/30 font-semibold text-xs transition flex items-center justify-center space-x-2"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download MP4 (-14 LUFS)</span>
          </a>
        </div>
      </div>
    </div>
  );
}
