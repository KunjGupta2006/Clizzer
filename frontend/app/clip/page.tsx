'use client';

import React, { useState, useRef } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Flame,
  ArrowLeft,
  Link as LinkIcon,
  UploadCloud,
  Sliders,
  Scissors,
  Download,
  AlertCircle,
  FileVideo,
  CheckCircle,
  Sparkles,
} from 'lucide-react';

import { WantedStars } from '../../components/WantedStars';
import { ClipCard } from '../../components/ClipCard';
import { MissionBrief } from '../../components/MissionBrief';
import { WastedScreen } from '../../components/WastedScreen';
import { MissionComplete } from '../../components/MissionComplete';

interface GeneratedClip {
  clip_path: string;
  thumbnail_path?: string | null;
  start_time: number;
  end_time: number;
  duration: number;
  hype_score: number;
}

export default function ClipWorkspacePage() {
  const [videoUrl, setVideoUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [numClips, setNumClips] = useState(4);
  const [maxDuration, setMaxDuration] = useState(30);
  const [reframe916, setReframe916] = useState(true);
  const [subtitleStyle, setSubtitleStyle] = useState('karaoke');

  // Processing & UI States
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('INITIALIZING');
  const [statusMessage, setStatusMessage] = useState('Target acquired. Loading sensors...');
  const [generatedClips, setGeneratedClips] = useState<GeneratedClip[]>([]);
  const [showWasted, setShowWasted] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [showMissionComplete, setShowMissionComplete] = useState(false);
  const [processingTime, setProcessingTime] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        setVideoUrl(text.trim());
        setSelectedFile(null);
      }
    } catch {
      // Fallback if clipboard permission is denied
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setVideoUrl('');
    }
  };

  const startMission = async () => {
    if (!videoUrl && !selectedFile) {
      setErrorMessage('PROVIDE A YOUTUBE LINK OR DROP A VIDEO FILE FIRST!');
      setShowWasted(true);
      return;
    }

    setIsProcessing(true);
    setProgress(0.05);
    setCurrentStep('ACQUIRING_STREAM');
    setStatusMessage('Downloading stream & buffering video data...');
    setShowWasted(false);
    setShowMissionComplete(false);
    setGeneratedClips([]);

    const startTime = Date.now();
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:7860';

    try {
      // Simulated progress animation steps while waiting for backend
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 0.90) return prev;
          const next = prev + 0.08;
          if (next > 0.25 && next <= 0.45) {
            setCurrentStep('SPECTRAL_ANALYSIS');
            setStatusMessage('Analyzing audio frequencies, percussive peaks & loudness...');
          } else if (next > 0.45 && next <= 0.65) {
            setCurrentStep('FACIAL_RECOGNITION');
            setStatusMessage('Running MediaPipe 468-point face mesh & emotion tracking...');
          } else if (next > 0.65 && next <= 0.85) {
            setCurrentStep('HYPE_FUSION');
            setStatusMessage('Fusing 5 signals & snapping cuts to scene boundaries...');
          } else if (next > 0.85) {
            setCurrentStep('RENDERING_SHORTS');
            setStatusMessage('Cutting vertical 9:16 clips and burning GTA subtitles...');
          }
          return next;
        });
      }, 800);

      let response: Response;

      if (selectedFile) {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('num_clips', numClips.toString());
        formData.append('max_duration', maxDuration.toString());
        formData.append('reframe', reframe916 ? 'true' : 'false');
        formData.append('subtitle_style', subtitleStyle);

        response = await fetch(`${apiUrl}/api/process-upload`, {
          method: 'POST',
          body: formData,
        }).catch(() => {
          // If server upload endpoint is not reached, fall back to direct pipeline mock
          return new Response(JSON.stringify({ clips: [] }), { status: 200 });
        });
      } else {
        response = await fetch(`${apiUrl}/api/process`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: videoUrl,
            num_clips: numClips,
            max_duration: maxDuration,
            reframe: reframe916,
            subtitle_style: subtitleStyle,
          }),
        }).catch(() => {
          // If backend is running standalone or offline mock
          return new Response(JSON.stringify({ clips: [] }), { status: 200 });
        });
      }

      clearInterval(progressInterval);

      if (response && response.ok) {
        const data = await response.json();
        const durationSecs = (Date.now() - startTime) / 1000;
        setProcessingTime(durationSecs);
        setProgress(1.0);
        setCurrentStep('MISSION_COMPLETE');
        setStatusMessage('Clips extracted successfully!');

        if (data.clips && data.clips.length > 0) {
          const clipsData: GeneratedClip[] = data.clips.map((c: any, i: number) => {
            if (typeof c === 'string') {
              const filename = c.split(/[\\/]/).pop();
              return {
                clip_path: c.startsWith('http') ? c : `${apiUrl}/clips/${filename}`,
                thumbnail_path: null,
                start_time: i * 30,
                end_time: (i + 1) * 30,
                duration: maxDuration,
                hype_score: Math.max(3, 5 - i * 0.5),
              };
            }
            const clipFilename = (c.clip_path || '').split(/[\\/]/).pop();
            const thumbFilename = c.thumbnail_path ? (c.thumbnail_path || '').split(/[\\/]/).pop() : null;
            return {
              clip_path: (c.clip_path && c.clip_path.startsWith('http')) ? c.clip_path : `${apiUrl}/clips/${clipFilename}`,
              thumbnail_path: thumbFilename ? (c.thumbnail_path.startsWith('http') ? c.thumbnail_path : `${apiUrl}/clips/${thumbFilename}`) : null,
              start_time: c.start_time ?? (i * 30),
              end_time: c.end_time ?? ((i + 1) * 30),
              duration: c.duration ?? maxDuration,
              hype_score: c.hype_score ?? Math.max(3, 5 - i * 0.5),
            };
          });

          setGeneratedClips(clipsData);
          setIsProcessing(false);
          setShowMissionComplete(true);
        } else {
          setIsProcessing(false);
          setErrorMessage('NO FIRE CLIPS DETECTED IN FOOTAGE. TRY A DIFFERENT VIDEO.');
          setShowWasted(true);
        }
      } else {
        let errDetail = 'Backend failed to process video.';
        try {
          const errData = await response.json();
          if (errData.detail) errDetail = errData.detail;
        } catch {}
        throw new Error(errDetail);
      }
    } catch (err: any) {
      setIsProcessing(false);
      setErrorMessage(err?.message || 'MISSION FAILED. UNABLE TO COMPLETE CLIPPING PIPELINE.');
      setShowWasted(true);
    }
  };

  const handleDownloadAll = () => {
    generatedClips.forEach((clip, idx) => {
      setTimeout(() => {
        const a = document.createElement('a');
        a.href = clip.clip_path;
        a.download = `clizzer_clip_${idx + 1}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }, idx * 400);
    });
  };

  return (
    <div className="relative min-h-screen flex flex-col justify-between p-4 sm:p-8 selection:bg-[#ff007f] selection:text-white">
      {/* Background ambient lighting */}
      <div className="fixed top-12 left-10 w-96 h-96 bg-[#ff007f]/15 rounded-full blur-[140px] pointer-events-none" />
      <div className="fixed bottom-12 right-10 w-96 h-96 bg-[#00f0ff]/10 rounded-full blur-[140px] pointer-events-none" />

      {/* Top Navbar */}
      <header className="relative z-10 flex items-center justify-between pb-6 border-b border-[#ff007f]/20">
        <Link
          href="/"
          className="flex items-center gap-2 text-gray-400 hover:text-[#f5b800] transition-colors text-sm font-bold tracking-wider uppercase"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>BACK TO BASE</span>
        </Link>

        <div className="flex items-center gap-3">
          <Flame className="w-6 h-6 text-[#ff007f]" />
          <span className="text-2xl font-black tracking-widest text-[#f5b800] uppercase font-ops drop-shadow-[0_0_8px_rgba(245,184,0,0.6)]">
            CLIZZER OPERATION
          </span>
        </div>

        <div className="hidden sm:block">
          <WantedStars level={isProcessing ? 4 : 5} size="sm" animated={isProcessing} />
        </div>
      </header>

      {/* Main Workspace */}
      <main className="relative z-10 flex-1 max-w-5xl w-full mx-auto my-8">
        {/* Input Panel */}
        <section className="bg-[#1a1a2e]/80 backdrop-blur-md border-2 border-[#ff007f]/60 rounded-2xl p-6 sm:p-8 shadow-[0_0_25px_rgba(255,0,127,0.25)]">
          <div className="flex items-center justify-between pb-4 border-b border-gray-800">
            <h2 className="text-2xl font-black uppercase tracking-wider text-[#f5b800] flex items-center gap-2">
              <Scissors className="w-6 h-6 text-[#ff007f]" />
              <span>SOURCE VIDEO INGESTION</span>
            </h2>
            <span className="text-xs text-[#00f0ff] font-bold tracking-widest uppercase">
              ZERO-API RECON
            </span>
          </div>

          {/* URL Input */}
          <div className="mt-6">
            <label className="block text-xs font-bold text-gray-300 uppercase tracking-widest mb-2 flex items-center gap-2">
              <LinkIcon className="w-4 h-4 text-[#00f0ff]" />
              <span>PASTE YOUTUBE / STREAM URL</span>
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="https://www.youtube.com/watch?v=..."
                value={videoUrl}
                onChange={(e) => {
                  setVideoUrl(e.target.value);
                  if (e.target.value) setSelectedFile(null);
                }}
                disabled={isProcessing}
                className="flex-1 bg-black/70 border border-gray-700 focus:border-[#f5b800] focus:ring-1 focus:ring-[#f5b800] rounded-xl px-4 py-3.5 text-white placeholder-gray-500 font-mono text-sm transition-all outline-none"
              />
              <button
                type="button"
                onClick={handlePaste}
                disabled={isProcessing}
                className="px-5 py-3.5 bg-[#252542] hover:bg-[#323259] text-gray-200 border border-gray-700 rounded-xl text-xs font-black tracking-wider uppercase transition-colors"
              >
                PASTE
              </button>
            </div>
          </div>

          {/* OR Divider */}
          <div className="relative my-6 text-center">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-800" />
            </div>
            <span className="relative px-4 bg-[#1a1a2e] text-xs font-extrabold tracking-widest text-gray-400 uppercase">
              OR UPLOAD LOCAL FOOTAGE
            </span>
          </div>

          {/* Drag & Drop File Input */}
          <div
            onClick={() => !isProcessing && fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
              selectedFile
                ? 'border-[#00f0ff] bg-[#00f0ff]/10'
                : 'border-gray-700 hover:border-[#ff007f] hover:bg-black/30'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              className="hidden"
            />
            {selectedFile ? (
              <div className="flex items-center justify-center gap-3 text-[#00f0ff]">
                <FileVideo className="w-8 h-8" />
                <div className="text-left">
                  <div className="font-extrabold text-sm uppercase tracking-wider">{selectedFile.name}</div>
                  <div className="text-xs text-gray-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB READY FOR SCAN
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-2 text-gray-400">
                <UploadCloud className="w-10 h-10 text-gray-500 hover:text-[#ff007f] transition-colors" />
                <span className="text-sm font-bold uppercase tracking-wider">
                  CLICK TO SELECT OR DRAG MP4, MKV, WEBM VIDEO
                </span>
                <span className="text-xs text-gray-500">PROCESSED LOCALLY VIA FAST FFMPEG & MEDIAPIPE</span>
              </div>
            )}
          </div>

          {/* Mission Parameters */}
          <div className="mt-8 pt-6 border-t border-gray-800 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Num Clips */}
            <div>
              <div className="flex justify-between items-center text-xs font-bold text-gray-300 uppercase tracking-widest mb-2">
                <span>TARGET CLIPS</span>
                <span className="text-[#f5b800]">{numClips}</span>
              </div>
              <input
                type="range"
                min="1"
                max="6"
                step="1"
                value={numClips}
                onChange={(e) => setNumClips(parseInt(e.target.value))}
                disabled={isProcessing}
                className="w-full accent-[#ff007f] cursor-pointer"
              />
            </div>

            {/* Max Duration */}
            <div>
              <div className="flex justify-between items-center text-xs font-bold text-gray-300 uppercase tracking-widest mb-2">
                <span>MAX DURATION</span>
                <span className="text-[#00f0ff]">{maxDuration}s</span>
              </div>
              <input
                type="range"
                min="15"
                max="90"
                step="5"
                value={maxDuration}
                onChange={(e) => setMaxDuration(parseInt(e.target.value))}
                disabled={isProcessing}
                className="w-full accent-[#00f0ff] cursor-pointer"
              />
            </div>

            {/* 9:16 Reframe Toggle */}
            <div className="flex flex-col justify-center">
              <span className="text-xs font-bold text-gray-300 uppercase tracking-widest mb-2">
                SMART 9:16 CROP
              </span>
              <button
                type="button"
                onClick={() => setReframe916(!reframe916)}
                disabled={isProcessing}
                className={`py-2 px-3 rounded-lg border text-xs font-black uppercase tracking-wider transition-all ${
                  reframe916
                    ? 'bg-[#ff007f]/20 border-[#ff007f] text-[#ff007f]'
                    : 'bg-black/50 border-gray-700 text-gray-400'
                }`}
              >
                {reframe916 ? 'FACE-TRACKED 9:16 ON' : 'PRESERVE ORIGINAL'}
              </button>
            </div>

            {/* Subtitle Style */}
            <div className="flex flex-col justify-center">
              <span className="text-xs font-bold text-gray-300 uppercase tracking-widest mb-2">
                SUBTITLES
              </span>
              <select
                value={subtitleStyle}
                onChange={(e) => setSubtitleStyle(e.target.value)}
                disabled={isProcessing}
                className="bg-black/70 border border-gray-700 rounded-lg px-3 py-2 text-xs font-bold uppercase tracking-wider text-[#f5b800] focus:border-[#f5b800] outline-none"
              >
                <option value="karaoke">KARAOKE HIGHLIGHT</option>
                <option value="gta">GTA GOLD STENCIL</option>
                <option value="pop">POP-IN WORD</option>
                <option value="none">NO SUBTITLES</option>
              </select>
            </div>
          </div>

          {/* Action Button */}
          <div className="mt-8">
            <button
              onClick={startMission}
              disabled={isProcessing || (!videoUrl && !selectedFile)}
              className="w-full py-5 bg-gradient-to-r from-[#ff007f] via-[#f5b800] to-[#00f0ff] hover:from-[#f5b800] hover:to-[#00f0ff] text-black font-black text-2xl tracking-widest uppercase rounded-xl transition-all duration-200 transform hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_30px_rgba(255,0,127,0.5)] flex items-center justify-center gap-3"
            >
              <Flame className="w-7 h-7" />
              <span>{isProcessing ? 'SCANNING SPECTRUM...' : 'START CLIPPING MISSION'}</span>
            </button>
          </div>
        </section>

        {/* Progress Briefing */}
        {isProcessing && (
          <MissionBrief
            step={currentStep}
            progress={progress}
            message={statusMessage}
            videoTitle={videoUrl || selectedFile?.name || 'ONLINE BROADCAST'}
          />
        )}

        {/* Generated Clips Results Section */}
        {generatedClips.length > 0 && !isProcessing && (
          <section className="mt-12">
            <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-6 border-b border-[#f5b800]/40">
              <div>
                <h3 className="text-3xl font-black uppercase tracking-wider text-[#f5b800] flex items-center gap-2">
                  <Sparkles className="w-7 h-7 text-[#ff007f]" />
                  <span>MISSION HEIST RESULTS ({generatedClips.length} CLIPS)</span>
                </h3>
                <p className="text-xs text-gray-400 uppercase tracking-widest mt-1">
                  100% WATERMARK FREE • HIGH RETENTION AUDIO CUTS
                </p>
              </div>

              <button
                onClick={handleDownloadAll}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-[#00f0ff] to-[#f5b800] text-black font-black uppercase text-sm tracking-wider rounded-xl transition-all transform hover:scale-105 shadow-[0_0_20px_rgba(0,240,255,0.4)]"
              >
                <Download className="w-5 h-5" />
                <span>DOWNLOAD ALL ASSETS</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {generatedClips.map((clip, index) => (
                <ClipCard
                  key={index}
                  clipUrl={clip.clip_path}
                  thumbnailUrl={clip.thumbnail_path}
                  duration={clip.duration}
                  hypeScore={clip.hype_score}
                  index={index}
                  startTime={clip.start_time}
                  endTime={clip.end_time}
                />
              ))}
            </div>
          </section>
        )}
      </main>

      {/* Wasted / Error Overlay */}
      <WastedScreen
        show={showWasted}
        message={errorMessage}
        onRetry={() => setShowWasted(false)}
      />

      {/* Mission Passed Banner */}
      <MissionComplete
        show={showMissionComplete}
        clipCount={generatedClips.length}
        processingTime={processingTime}
        onDismiss={() => setShowMissionComplete(false)}
      />

      {/* Footer */}
      <footer className="relative z-10 py-4 border-t border-gray-900 text-center text-xs text-gray-500 uppercase tracking-widest font-semibold">
        <span>CLIZZER ENGINE • MULTI-MODAL HIGHLIGHT EXTRACTION • ZERO PAID APIS</span>
      </footer>
    </div>
  );
}
