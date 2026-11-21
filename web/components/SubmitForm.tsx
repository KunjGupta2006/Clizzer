"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { createJob } from "@/lib/api";
import { UploadCloud, Link as LinkIcon, Film, Sparkles, AlertCircle, Camera, Layers, Sliders, CheckCircle2, Wand2 } from "lucide-react";

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

  // Stream Dual-Frame Layout States
  const [layout, setLayout] = useState("single"); // "single", "game_top", "cam_top", "pip"
  const [camX, setCamX] = useState(72);
  const [camY, setCamY] = useState(5);
  const [camW, setCamW] = useState(26);
  const [camH, setCamH] = useState(32);

  const [gameX, setGameX] = useState(0);
  const [gameY, setGameY] = useState(0);
  const [gameW, setGameW] = useState(72);
  const [gameH, setGameH] = useState(100);

  const applyCamPreset = (preset: "top-right" | "top-left" | "bottom-right" | "bottom-left") => {
    if (preset === "top-right") {
      setCamX(72);
      setCamY(5);
      setCamW(26);
      setCamH(32);
      // Auto-exclude from game view
      setGameX(0);
      setGameY(0);
      setGameW(72);
      setGameH(100);
    } else if (preset === "top-left") {
      setCamX(2);
      setCamY(5);
      setCamW(26);
      setCamH(32);
      // Auto-exclude from game view
      setGameX(28);
      setGameY(0);
      setGameW(72);
      setGameH(100);
    } else if (preset === "bottom-right") {
      setCamX(72);
      setCamY(63);
      setCamW(26);
      setCamH(32);
      setGameX(0);
      setGameY(0);
      setGameW(72);
      setGameH(100);
    } else if (preset === "bottom-left") {
      setCamX(2);
      setCamY(63);
      setCamW(26);
      setCamH(32);
      setGameX(28);
      setGameY(0);
      setGameW(72);
      setGameH(100);
    }
  };

  const applyGamePreset = (preset: "full" | "exclude-right" | "exclude-left" | "center") => {
    if (preset === "full") {
      setGameX(0);
      setGameY(0);
      setGameW(100);
      setGameH(100);
    } else if (preset === "exclude-right") {
      setGameX(0);
      setGameY(0);
      setGameW(72);
      setGameH(100);
    } else if (preset === "exclude-left") {
      setGameX(28);
      setGameY(0);
      setGameW(72);
      setGameH(100);
    } else if (preset === "center") {
      setGameX(14);
      setGameY(0);
      setGameW(72);
      setGameH(100);
    }
  };

  const autoExcludeCam = () => {
    if (camX >= 50) {
      setGameX(0);
      setGameY(0);
      setGameW(Math.max(40, camX));
      setGameH(100);
    } else {
      const startX = Math.min(60, camX + camW);
      setGameX(startX);
      setGameY(0);
      setGameW(100 - startX);
      setGameH(100);
    }
  };

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

    const regions =
      layout !== "single"
        ? [
            {
              label: "cam",
              x: camX / 100,
              y: camY / 100,
              w: camW / 100,
              h: camH / 100,
            },
            {
              label: "game",
              x: gameX / 100,
              y: gameY / 100,
              w: gameW / 100,
              h: gameH / 100,
            },
          ]
        : undefined;

    setLoading(true);
    try {
      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("num_clips", numClips.toString());
        formData.append("min_len", minLen.toString());
        formData.append("max_len", maxLen.toString());
        formData.append("aspect", aspect);
        formData.append("layout", layout);
        if (regions) {
          formData.append("regions", JSON.stringify(regions));
        }
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
          layout,
          regions,
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
                {file ? file.name : "Drop MP4, MKV, or WebM here, or browse (multi-hour streams & large files supported)"}
              </p>
              <p className="text-xs text-gray-500 mt-1">Multi-gigabyte uploads streamed direct-to-disk • 3+ hour videos accelerated</p>
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
              Stream Frame Layout
            </label>
            <select
              value={layout}
              onChange={(e) => setLayout(e.target.value)}
              className="w-full bg-[#0e0f15] border border-pink-500/50 rounded-xl px-4 py-2 text-sm text-pink-300 font-medium focus:outline-none focus:border-pink-500"
            >
              <option value="single">Standard (Full Screen / Blur Letterbox)</option>
              <option value="game_top">Dual Frame: Game Top, Streamer Bottom</option>
              <option value="cam_top">Dual Frame: Streamer Top, Game Bottom</option>
              <option value="pip">Picture-in-Picture (PiP Cam Overlay)</option>
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

          <div className="flex space-x-2 sm:col-span-2">
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

        {/* Dual Frame Interactive Region Selector */}
        {layout !== "single" && (
          <div className="p-4 rounded-xl bg-[#0a0a0f] border border-pink-500/30 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-pink-400">
                <Layers className="w-4 h-4" />
                <span className="text-xs font-mono font-bold uppercase tracking-wider">
                  Stream Dual-Frame Calibration
                </span>
              </div>
              <button
                type="button"
                onClick={autoExcludeCam}
                className="text-[11px] text-pink-400 hover:text-pink-300 font-mono flex items-center space-x-1 bg-pink-500/10 px-2 py-1 rounded border border-pink-500/20"
              >
                <Wand2 className="w-3 h-3" />
                <span>AI Auto-Exclude Cam from Game</span>
              </button>
            </div>

            {/* Visual Stream Mockup */}
            <div className="relative aspect-video w-full rounded-lg bg-[#14151d] border border-gray-800 overflow-hidden shadow-inner">
              {/* Gameplay Region */}
              <div
                style={{
                  left: `${gameX}%`,
                  top: `${gameY}%`,
                  width: `${gameW}%`,
                  height: `${gameH}%`,
                }}
                className="absolute border-2 border-dashed border-pink-500/70 bg-pink-500/10 flex items-center justify-center transition-all"
              >
                <span className="text-[10px] font-mono font-bold text-pink-300 bg-black/70 px-2 py-0.5 rounded">
                  🎮 Game View ({gameW}×{gameH}%)
                </span>
              </div>

              {/* Streamer Cam Region */}
              <div
                style={{
                  left: `${camX}%`,
                  top: `${camY}%`,
                  width: `${camW}%`,
                  height: `${camH}%`,
                }}
                className="absolute border-2 border-solid border-cyan-400 bg-cyan-400/20 flex items-center justify-center shadow-lg transition-all"
              >
                <span className="text-[10px] font-mono font-bold text-cyan-200 bg-black/80 px-1.5 py-0.5 rounded flex items-center space-x-1">
                  <Camera className="w-3 h-3 text-cyan-400" />
                  <span>Streamer Cam</span>
                </span>
              </div>
            </div>

            {/* Gameplay Presets */}
            <div className="space-y-1.5">
              <label className="block text-[11px] font-mono text-gray-400 uppercase tracking-wider">
                1. Gameplay Region (Exclude Cam to Avoid Duplicate Face)
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => applyGamePreset("exclude-right")}
                  className={`py-1.5 px-3 rounded-lg border font-mono transition ${
                    gameW < 100 && gameX === 0
                      ? "bg-pink-500/20 border-pink-400 text-pink-300"
                      : "bg-[#14151d] border-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  Exclude Right Cam
                </button>
                <button
                  type="button"
                  onClick={() => applyGamePreset("exclude-left")}
                  className={`py-1.5 px-3 rounded-lg border font-mono transition ${
                    gameX > 0
                      ? "bg-pink-500/20 border-pink-400 text-pink-300"
                      : "bg-[#14151d] border-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  Exclude Left Cam
                </button>
                <button
                  type="button"
                  onClick={() => applyGamePreset("center")}
                  className={`py-1.5 px-3 rounded-lg border font-mono transition ${
                    gameX > 0 && gameW < 80
                      ? "bg-pink-500/20 border-pink-400 text-pink-300"
                      : "bg-[#14151d] border-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  Center Gameplay
                </button>
                <button
                  type="button"
                  onClick={() => applyGamePreset("full")}
                  className={`py-1.5 px-3 rounded-lg border font-mono transition ${
                    gameW === 100 && gameX === 0
                      ? "bg-pink-500/20 border-pink-400 text-pink-300"
                      : "bg-[#14151d] border-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  Full Screen
                </button>
              </div>
            </div>

            {/* Quick Presets for Streamer Camera */}
            <div className="space-y-1.5">
              <label className="block text-[11px] font-mono text-gray-400 uppercase tracking-wider">
                2. Streamer Webcam Position
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => applyCamPreset("top-right")}
                  className={`py-1.5 px-3 rounded-lg border font-mono transition ${
                    camX >= 50 && camY < 50
                      ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                      : "bg-[#14151d] border-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  Top-Right
                </button>
                <button
                  type="button"
                  onClick={() => applyCamPreset("top-left")}
                  className={`py-1.5 px-3 rounded-lg border font-mono transition ${
                    camX < 50 && camY < 50
                      ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                      : "bg-[#14151d] border-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  Top-Left
                </button>
                <button
                  type="button"
                  onClick={() => applyCamPreset("bottom-right")}
                  className={`py-1.5 px-3 rounded-lg border font-mono transition ${
                    camX >= 50 && camY >= 50
                      ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                      : "bg-[#14151d] border-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  Bottom-Right
                </button>
                <button
                  type="button"
                  onClick={() => applyCamPreset("bottom-left")}
                  className={`py-1.5 px-3 rounded-lg border font-mono transition ${
                    camX < 50 && camY >= 50
                      ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                      : "bg-[#14151d] border-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  Bottom-Left
                </button>
              </div>
            </div>

            {/* Fine Tuning Sliders: Streamer Camera */}
            <div className="pt-2 border-t border-gray-800/80">
              <span className="text-[11px] font-mono text-cyan-400 block mb-2 font-semibold">
                Adjust Streamer Cam Box
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px] font-mono text-gray-400">
                <div>
                  <span>Cam X: {camX}%</span>
                  <input
                    type="range"
                    min={0}
                    max={80}
                    value={camX}
                    onChange={(e) => setCamX(parseInt(e.target.value) || 0)}
                    className="w-full accent-cyan-400"
                  />
                </div>
                <div>
                  <span>Cam Y: {camY}%</span>
                  <input
                    type="range"
                    min={0}
                    max={75}
                    value={camY}
                    onChange={(e) => setCamY(parseInt(e.target.value) || 0)}
                    className="w-full accent-cyan-400"
                  />
                </div>
                <div>
                  <span>Cam W: {camW}%</span>
                  <input
                    type="range"
                    min={15}
                    max={50}
                    value={camW}
                    onChange={(e) => setCamW(parseInt(e.target.value) || 26)}
                    className="w-full accent-cyan-400"
                  />
                </div>
                <div>
                  <span>Cam H: {camH}%</span>
                  <input
                    type="range"
                    min={15}
                    max={60}
                    value={camH}
                    onChange={(e) => setCamH(parseInt(e.target.value) || 32)}
                    className="w-full accent-cyan-400"
                  />
                </div>
              </div>
            </div>

            {/* Fine Tuning Sliders: Gameplay Box */}
            <div className="pt-2 border-t border-gray-800/80">
              <span className="text-[11px] font-mono text-pink-400 block mb-2 font-semibold">
                Adjust Gameplay Box
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px] font-mono text-gray-400">
                <div>
                  <span>Game X: {gameX}%</span>
                  <input
                    type="range"
                    min={0}
                    max={60}
                    value={gameX}
                    onChange={(e) => setGameX(parseInt(e.target.value) || 0)}
                    className="w-full accent-pink-400"
                  />
                </div>
                <div>
                  <span>Game Y: {gameY}%</span>
                  <input
                    type="range"
                    min={0}
                    max={50}
                    value={gameY}
                    onChange={(e) => setGameY(parseInt(e.target.value) || 0)}
                    className="w-full accent-pink-400"
                  />
                </div>
                <div>
                  <span>Game W: {gameW}%</span>
                  <input
                    type="range"
                    min={40}
                    max={100}
                    value={gameW}
                    onChange={(e) => setGameW(parseInt(e.target.value) || 72)}
                    className="w-full accent-pink-400"
                  />
                </div>
                <div>
                  <span>Game H: {gameH}%</span>
                  <input
                    type="range"
                    min={40}
                    max={100}
                    value={gameH}
                    onChange={(e) => setGameH(parseInt(e.target.value) || 100)}
                    className="w-full accent-pink-400"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

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
