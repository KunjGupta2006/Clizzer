'use client';

import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Award, Zap, CheckCircle2 } from 'lucide-react';

export interface MissionCompleteProps {
  show: boolean;
  clipCount?: number;
  processingTime?: number;
  onDismiss?: () => void;
}

export const MissionComplete: React.FC<MissionCompleteProps> = ({
  show,
  clipCount = 4,
  processingTime = 12.5,
  onDismiss,
}) => {
  useEffect(() => {
    if (show && onDismiss) {
      const timer = setTimeout(() => {
        onDismiss();
      }, 3500);
      return () => clearTimeout(timer);
    }
  }, [show, onDismiss]);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key="mission-complete-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm pointer-events-none select-none"
        >
          {/* Green flash background */}
          <motion.div
            initial={{ opacity: 0.8 }}
            animate={{ opacity: 0 }}
            transition={{ duration: 0.7 }}
            className="absolute inset-0 bg-[#10b981]/25 pointer-events-none"
          />

          <motion.div
            initial={{ scale: 0.7, opacity: 0, y: 30 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 1.1, opacity: 0 }}
            transition={{ type: 'spring', damping: 15, stiffness: 180 }}
            className="relative z-10 text-center flex flex-col items-center"
          >
            {/* Mission Passed Header */}
            <div className="flex items-center gap-3 mb-2">
              <Award className="w-10 h-10 text-[#f5b800] animate-bounce" />
            </div>

            <h1 className="text-5xl sm:text-8xl font-black tracking-widest text-[#10b981] drop-shadow-[0_0_35px_rgba(16,185,129,0.9)] font-ops uppercase">
              MISSION PASSED!
            </h1>

            {/* RESPECT + Subtext */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: [1, 1.15, 1] }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="mt-3 text-3xl sm:text-5xl font-black tracking-widest text-[#f5b800] drop-shadow-[0_0_20px_rgba(245,184,0,0.8)] uppercase font-gta"
            >
              RESPECT +
            </motion.div>

            {/* Stats pill */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-6 flex items-center gap-6 px-6 py-2.5 bg-black/80 border border-[#10b981]/50 rounded-full text-sm font-bold tracking-wider text-gray-200"
            >
              <span className="flex items-center gap-1.5 text-[#00f0ff]">
                <CheckCircle2 className="w-4 h-4" />
                {clipCount} HIGHLIGHTS CUT
              </span>
              <span className="w-1 h-1 bg-gray-600 rounded-full" />
              <span className="flex items-center gap-1.5 text-[#f5b800]">
                <Zap className="w-4 h-4" />
                {processingTime.toFixed(1)}s TOTAL TIME
              </span>
            </motion.div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
