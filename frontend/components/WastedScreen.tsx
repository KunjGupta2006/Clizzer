'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, AlertTriangle } from 'lucide-react';

export interface WastedScreenProps {
  show: boolean;
  message?: string;
  onRetry?: () => void;
}

export const WastedScreen: React.FC<WastedScreenProps> = ({
  show,
  message = 'MISSION COMPROMISED. TARGET FOOTAGE FAILED TO PROCESS.',
  onRetry,
}) => {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key="wasted-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/85 backdrop-blur-md p-6 select-none"
        >
          {/* Classic GTA Wasted Vignette Red Tint */}
          <div className="absolute inset-0 bg-red-950/40 pointer-events-none mix-blend-multiply" />

          {/* WASTED Text */}
          <motion.div
            initial={{ scale: 0.8, opacity: 0, filter: 'blur(10px)' }}
            animate={{ scale: [0.8, 1.05, 1], opacity: 1, filter: 'blur(0px)' }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className="relative z-10 text-center"
          >
            <h1 className="text-7xl sm:text-9xl font-black tracking-widest text-[#e11d48] drop-shadow-[0_0_35px_rgba(225,29,72,0.85)] font-ops uppercase">
              WASTED
            </h1>

            <p className="mt-4 max-w-md mx-auto text-sm sm:text-base text-gray-300 font-semibold tracking-wider uppercase flex items-center justify-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0" />
              <span>{message}</span>
            </p>

            {onRetry && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onRetry}
                className="mt-8 inline-flex items-center gap-2 px-8 py-3.5 bg-gradient-to-r from-red-600 to-[#ff007f] hover:from-[#ff007f] hover:to-[#f5b800] text-white font-black text-lg tracking-widest uppercase rounded-xl shadow-[0_0_20px_rgba(225,29,72,0.6)] transition-all duration-200"
              >
                <RotateCcw className="w-5 h-5" />
                <span>TRY AGAIN</span>
              </motion.button>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
