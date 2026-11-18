'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Radio, Crosshair, Cpu, Zap, Activity } from 'lucide-react';
import { WantedStars } from './WantedStars';

export interface MissionBriefProps {
  step: string;
  progress: number; // 0 to 1
  message: string;
  videoTitle?: string;
  wantedLevel?: number;
}

export const MissionBrief: React.FC<MissionBriefProps> = ({
  step,
  progress,
  message,
  videoTitle = 'TARGET FOOTAGE',
  wantedLevel = 3,
}) => {
  const percentage = Math.max(0, Math.min(100, Math.round(progress * 100)));
  const calculatedStars = Math.min(5, Math.max(1, Math.ceil(progress * 5)));

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      className="w-full max-w-2xl mx-auto my-6 p-6 bg-[#1a1a2e]/90 backdrop-blur-md border-2 border-[#ff007f] rounded-2xl shadow-[0_0_30px_rgba(255,0,127,0.3)] relative overflow-hidden"
    >
      {/* Background HUD Grid Line Accent */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-[#ff007f] via-[#f5b800] to-[#00f0ff] animate-pulse" />

      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#ff007f]/30">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Radio className="w-6 h-6 text-[#ff007f] animate-ping absolute opacity-75" />
            <Radio className="w-6 h-6 text-[#ff007f] relative" />
          </div>
          <div>
            <h3 className="text-xl font-black uppercase tracking-widest text-[#f5b800] drop-shadow-[0_0_8px_rgba(245,184,0,0.5)]">
              MISSION IN PROGRESS
            </h3>
            <p className="text-xs text-[#00f0ff] tracking-wider uppercase font-semibold">
              OPERATION: {step.replace(/_/g, ' ') || 'ANALYZING SPECTRUM'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-black/60 px-3 py-1.5 rounded-lg border border-[#f5b800]/40">
          <WantedStars level={calculatedStars} size="sm" animated={true} />
        </div>
      </div>

      {/* Target Title */}
      <div className="my-5 p-3.5 bg-black/50 border border-gray-800 rounded-lg flex items-start gap-3">
        <Crosshair className="w-5 h-5 text-[#00f0ff] shrink-0 mt-0.5" />
        <div className="overflow-hidden">
          <div className="text-[11px] text-gray-400 uppercase tracking-widest font-bold">
            OBJECTIVE TARGET
          </div>
          <div className="text-sm font-bold text-white truncate max-w-md">
            {videoTitle}
          </div>
        </div>
      </div>

      {/* Progress Bar styled as GTA Loading Bar */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs tracking-wider uppercase font-bold">
          <span className="text-gray-300 flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-[#ff007f] animate-spin" />
            {message || 'Scanning frequencies and facial geometry...'}
          </span>
          <span className="text-[#f5b800] text-sm font-black">
            {percentage}%
          </span>
        </div>

        <div className="h-4 w-full bg-black/80 rounded-full overflow-hidden p-0.5 border border-[#ff007f]/50">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-[#ff007f] via-[#f5b800] to-[#00f0ff]"
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ ease: 'easeOut', duration: 0.3 }}
          />
        </div>
      </div>

      {/* Sub-hud status icons */}
      <div className="grid grid-cols-3 gap-2 mt-5 pt-4 border-t border-gray-800/80 text-[11px] text-gray-400 font-semibold tracking-wider text-center">
        <div className="flex items-center justify-center gap-1.5">
          <Zap className="w-3.5 h-3.5 text-[#f5b800]" />
          <span>AUDIO PEAKS</span>
        </div>
        <div className="flex items-center justify-center gap-1.5 border-x border-gray-800">
          <Cpu className="w-3.5 h-3.5 text-[#00f0ff]" />
          <span>FACE MESH</span>
        </div>
        <div className="flex items-center justify-center gap-1.5">
          <Crosshair className="w-3.5 h-3.5 text-[#ff007f]" />
          <span>SCENE CUTS</span>
        </div>
      </div>
    </motion.div>
  );
};
