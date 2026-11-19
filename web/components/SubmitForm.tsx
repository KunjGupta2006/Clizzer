"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { createJob } from "@/lib/api";
import { UploadCloud, Link as LinkIcon, Film, Sparkles, AlertCircle } from "lucide-react";

function extractYouTubeId(url: string): string | null {
  const match = url.match(
    /(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=|shorts\/))([\w-]{11})/
  );
  return match ? match[1] : null;
}

export default function SubmitForm() {
  const router = useRouter();
  const [sourceUrl, setSourceUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [numClips, setNumClips] = useState(5);
  const [minLen, setMinLen] = useState(20);
  const [maxLen, setMaxLen] = useState(60);
  const [aspect, setAspect] = useState("9:16");
  const [mode, setMode] = useState("auto");
  const [keywords, setKeywords] = useState("GTA 6, Vice City, Lucia, Jason");
  const [captions, setCaptions] = useState(true);
  const [permissionConfirmed, setPermissionConfirmed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!permissionConfirmed) {
      setError("You must confirm that you have permission to clip this video.");
      return;
    }

    if (!sourceUrl && !file) {
      setError("Please provide a YouTube URL or upload a video file.");
      return;
    }

    setLoading(true);
    try {
      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("num_clips", numClips.toString());
        formData.append("min_len", minLen.toString());
        formData.append("max_len", maxLen.toString());
        formData.append("aspect", aspect);
        formData.append("mode", mode);
        formData.append("keywords", keywords);
        formData.append("captions", captions.toString());
        formData.append("permission_confirmed", "true");

        const result = await createJob(formData);
        router.push(`/jobs/${result.job_id}`);
      } else {
        const result = await createJob({
          source_url: sourceUrl,
          num_clips: numClips,
          min_len: minLen,
          max_len: maxLen,
          aspect,
          mode,
          keywords: keywords.split(",").map((k) => k.trim()),
          captions,
          permission_confirmed: true,
        });
        router.push(`/jobs/${result.job_id}`);
      }
    } catch (err: any) {
      setError(err.message || "Failed to create clipping job.");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto bg-[#14151d] border border-gray-800 rounded-2xl p-8 shadow-2xl">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
          Transform Streams into <span className="text-pink-500">Viral Shorts</span>
        </h1>
        <p className="mt-2 text-sm text-gray-400">
          Multimodal AI analysis using audio acoustics, facial expressions, live chat excitement, and LLM hooks.
        </p>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 flex items-center space-x-3 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Input Method */}
        <div className="space-y-4">
          <label className="block text-xs font-mono uppercase tracking-wider text-gray-400">
            Source Video
          </label>
          <div className="flex rounded-xl bg-[#0e0f15] border border-gray-800 p-1">
            <input
              type="url"
              placeholder="Paste YouTube or Twitch link (e.g. https://www.youtube.com/watch?v=...)"
              value={sourceUrl}
              onChange={(e) => {
                setSourceUrl(e.target.value);
                setFile(null);
              }}
              className="flex-1 bg-transparent px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none"
            />
          </div>

          <div className="relative flex py-1 items-center">
            <div className="flex-grow border-t border-gray-800"></div>
            <span className="flex-shrink mx-4 text-xs font-mono text-gray-500 uppercase">Or upload file</span>
            <div className="flex-grow border-t border-gray-800"></div>
          </div>

          <div className="border-2 border-dashed border-gray-800 hover:border-pink-500/50 rounded-xl p-6 text-center transition cursor-pointer bg-[#0e0f15]/50">
            <input
              type="file"
              accept="video/*"
              onChange={(e) => {
                if (e.target.files?.[0]) {
                  setFile(e.target.files[0]);
                  setSourceUrl("");
                }
              }}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <UploadCloud className="mx-auto h-8 w-8 text-pink-500 mb-2" />
              <p className="text-sm font-medium text-gray-300">
                {file ? file.name : "Drop MP4, MKV, or WebM here, or browse"}
              </p>
              <p className="text-xs text-gray-500 mt-1">Up to 2GB file size</p>
            </label>
          </div>
          {/* Live Preview Container */}
          {sourceUrl && extractYouTubeId(sourceUrl) && (
            <div className="mt-3 bg-[#0a0a0f] border border-gray-800 rounded-xl p-3 shadow-inner">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-mono text-pink-400 uppercase font-semibold flex items-center space-x-1.5">
                  <Film className="w-3.5 h-3.5" />
                  <span>Live URL Preview</span>
                </span>
                <span className="text-[10px] text-gray-500 font-mono">YouTube</span>
              </div>
              <div className="relative aspect-video rounded-lg overflow-hidden border border-gray-800 bg-black">
                <iframe
                  src={`https://www.youtube-nocookie.com/embed/${extractYouTubeId(sourceUrl)}`}
                  title="Live YouTube Preview"
                  className="w-full h-full border-0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
            </div>
          )}

          {file && (
            <div className="mt-3 bg-[#0a0a0f] border border-gray-800 rounded-xl p-3 shadow-inner">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-mono text-cyan-400 uppercase font-semibold flex items-center space-x-1.5">
                  <Film className="w-3.5 h-3.5" />
                  <span>Local Video Preview</span>
                </span>
                <span className="text-[10px] text-gray-500 font-mono truncate max-w-[200px]">{file.name}</span>
              </div>
              <div className="relative aspect-video rounded-lg overflow-hidden border border-gray-800 bg-black">
                <video
                  src={URL.createObjectURL(file)}
                  controls
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}
        </div>

        {/* Configuration Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
          <div>
            <label className="block text-xs font-mono uppercase tracking-wider text-gray-400 mb-1.5">
              Target Mode
            </label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="w-full bg-[#0e0f15] border border-gray-800 rounded-xl px-4 py-2 text-sm text-white focus:outline-none focus:border-pink-500"
            >
              <option value="auto">Auto (Balanced Weights)</option>
              <option value="hype">Hype (Audio & Chat Spikes)</option>
              <option value="funny">Funny (Chat & Quotes)</option>
              <option value="informational">Informational (Dialogue & Text)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-mono uppercase tracking-wider text-gray-400 mb-1.5">
              Aspect Ratio
            </label>
            <select
              value={aspect}
              onChange={(e) => setAspect(e.target.value)}
              className="w-full bg-[#0e0f15] border border-gray-800 rounded-xl px-4 py-2 text-sm text-white focus:outline-none focus:border-pink-500"
            >
              <option value="9:16">9:16 (TikTok, Shorts, Reels)</option>
              <option value="1:1">1:1 (Square Feed)</option>
              <option value="16:9">16:9 (Landscape)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-mono uppercase tracking-wider text-gray-400 mb-1.5">
              Top Clips Count
            </label>
            <input
              type="number"
              min={1}
              max={30}
              value={numClips}
              onChange={(e) => setNumClips(parseInt(e.target.value) || 5)}
              className="w-full bg-[#0e0f15] border border-gray-800 rounded-xl px-4 py-2 text-sm text-white focus:outline-none focus:border-pink-500"
            />
          </div>

          <div className="flex space-x-2">
            <div className="flex-1">
              <label className="block text-xs font-mono uppercase tracking-wider text-gray-400 mb-1.5">
                Min Sec
              </label>
              <input
                type="number"
                min={5}
                max={60}
                value={minLen}
                onChange={(e) => setMinLen(parseInt(e.target.value) || 20)}
                className="w-full bg-[#0e0f15] border border-gray-800 rounded-xl px-4 py-2 text-sm text-white focus:outline-none focus:border-pink-500"
              />
            </div>
            <div className="flex-1">
              <label className="block text-xs font-mono uppercase tracking-wider text-gray-400 mb-1.5">
                Max Sec
              </label>
              <input
                type="number"
                min={20}
                max={120}
                value={maxLen}
                onChange={(e) => setMaxLen(parseInt(e.target.value) || 60)}
                className="w-full bg-[#0e0f15] border border-gray-800 rounded-xl px-4 py-2 text-sm text-white focus:outline-none focus:border-pink-500"
              />
            </div>
          </div>
        </div>

        {/* Keywords */}
        <div>
          <label className="block text-xs font-mono uppercase tracking-wider text-gray-400 mb-1.5">
            Boost Keywords (comma separated)
          </label>
          <input
            type="text"
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            className="w-full bg-[#0e0f15] border border-gray-800 rounded-xl px-4 py-2 text-sm text-white focus:outline-none focus:border-pink-500 font-mono text-xs"
          />
        </div>

        {/* Captions Toggle */}
        <div className="flex items-center space-x-3 p-3 rounded-xl bg-[#0e0f15] border border-gray-800">
          <input
            type="checkbox"
            id="captions-toggle"
            checked={captions}
            onChange={(e) => setCaptions(e.target.checked)}
            className="w-4 h-4 rounded border-gray-700 text-pink-500 focus:ring-0"
          />
          <label htmlFor="captions-toggle" className="text-sm text-gray-300 cursor-pointer">
            Burn animated karaoke ASS captions with active word highlight
          </label>
        </div>

        {/* Permission Confirmation Checkbox (Required) */}
        <div className="flex items-start space-x-3 p-4 rounded-xl bg-pink-500/5 border border-pink-500/20">
          <input
            type="checkbox"
            id="permission-checkbox"
            checked={permissionConfirmed}
            onChange={(e) => setPermissionConfirmed(e.target.checked)}
            className="mt-1 w-4 h-4 rounded border-gray-700 text-pink-500 focus:ring-0"
            required
          />
          <label htmlFor="permission-checkbox" className="text-xs text-gray-300 leading-relaxed cursor-pointer">
            <span className="font-semibold text-pink-400">Copyright & Terms Confirmation:</span> I confirm that I own or have obtained lawful authorization to process, analyze, and generate derivative clips from this video content.
          </label>
        </div>

        <button
          type="submit"
          disabled={loading || !permissionConfirmed}
          className="w-full py-3.5 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg shadow-pink-500/20 flex items-center justify-center space-x-2"
        >
          {loading ? (
            <span>Analyzing Multimodal Signals...</span>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Generate Viral Clips</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
