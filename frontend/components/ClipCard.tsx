'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Download, Play, Flame, Clock } from 'lucide-react';
import { WantedStars } from './WantedStars';

export interface ClipCardProps {
  clipUrl: string;
  thumbnailUrl?: string | null;
  duration: number;
  hypeScore: number;
  index: number;
  startTime?: number;
  endTime?: number;
  onDownload?: (clipUrl: string, index: number) => void;
}

export const ClipCard: React.FC<ClipCardProps> = ({
  clipUrl,
  thumbnailUrl,
  duration,
  hypeScore,
  index,
  startTime,
  endTime,
  onDownload,
}) => {
  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainingSecs = Math.floor(secs % 60);
    return `${mins}:${remainingSecs.toString().padStart(2, '0')}`;
  };

  const handleDownloadClick = () => {
    if (onDownload) {
      onDownload(clipUrl, index);
    } else {
      const a = document.createElement('a');
      a.href = clipUrl;
      a.download = `clizzer_clip_${index + 1}.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      className="group relative flex flex-col bg-[#1a1a2e] border-2 border-[#ff007f]/50 hover:border-[#f5b800] rounded-xl overflow-hidden transition-all duration-300 hover:shadow-[0_0_25px_rgba(255,0,127,0.35)]"
    >
      {/* Top Banner with Clip Number & Hype Score */}
      <div className="flex items-center justify-between px-4 py-2 bg-black/60 border-b border-[#ff007f]/30">
        <div className="flex items-center gap-2">
          <Flame className="w-5 h-5 text-[#ff007f] animate-pulse" />
          <span className="font-extrabold tracking-wider text-[#f5b800] text-lg uppercase">
            TARGET #{index + 1}
          </span>
        </div>
        <WantedStars level={hypeScore} size="sm" animated={true} />
      </div>

      {/* Video Preview Container */}
      <div className="relative aspect-[9/16] max-h-[460px] w-full bg-black/80 flex items-center justify-center overflow-hidden">
        <video
          src={clipUrl}
          poster={thumbnailUrl || undefined}
          controls
          preload="metadata"
          className="w-full h-full object-contain"
        />

        {/* Duration badge */}
        <div className="absolute bottom-3 right-3 flex items-center gap-1 px-2.5 py-1 bg-black/80 backdrop-blur-sm border border-[#00f0ff]/40 rounded text-xs text-[#00f0ff] font-bold">
          <Clock className="w-3.5 h-3.5" />
          <span>{duration.toFixed(1)}s</span>
        </div>

        {/* Timestamps if provided */}
        {startTime !== undefined && endTime !== undefined && (
          <div className="absolute top-3 left-3 px-2 py-0.5 bg-black/75 backdrop-blur-sm rounded text-[11px] text-gray-300 font-mono">
            {formatTime(startTime)} - {formatTime(endTime)}
          </div>
        )}
      </div>

      {/* Action Footer */}
      <div className="p-3 bg-[#111122] flex items-center justify-between border-t border-[#ff007f]/20">
        <div className="flex flex-col">
          <span className="text-[11px] text-gray-400 uppercase tracking-widest font-semibold">
            Status
          </span>
          <span className="text-sm font-black text-[#00f0ff] uppercase tracking-wider">
            FIRE SCORE {(hypeScore * 20).toFixed(0)}%
          </span>
        </div>

        <button
          onClick={handleDownloadClick}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-[#ff007f] to-[#f5b800] hover:from-[#f5b800] hover:to-[#00f0ff] text-black font-black uppercase text-sm tracking-wider rounded-lg transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-[0_0_12px_rgba(255,0,127,0.4)]"
        >
          <Download className="w-4 h-4" />
          <span>HEIST</span>
        </button>
      </div>
    </motion.div>
  );
};
