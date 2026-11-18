'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface GTALoaderProps {
  /** Controls overlay visibility */
  isLoading: boolean;
  /** Current progress percentage (0 - 100) */
  progress: number;
  /** Status or step message displayed beneath the title */
  message: string;
}

const letterVariants = {
  hidden: { opacity: 0, y: 6 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.1,
      duration: 0.2,
      repeat: Infinity,
      repeatDelay: 2.2,
    },
  }),
};

export const GTALoader: React.FC<GTALoaderProps> = ({
  isLoading,
  progress,
  message,
}) => {
  const clampedProgress = Math.max(0, Math.min(100, Math.round(progress)));
  const loadingCharacters = ['L', 'O', 'A', 'D', 'I', 'N', 'G', '.', '.', '.'];

  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          key="gta-loader-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          className="fixed inset-0 z-50 flex flex-col items-center justify-between bg-black/85 backdrop-blur-md text-white select-none overflow-hidden"
          role="status"
          aria-live="polite"
          aria-busy={isLoading}
        >
          {/* Subtle Ambient Vignette & Neon Glows */}
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(236,72,153,0.12)_0%,rgba(0,0,0,0.85)_80%)]" />

          {/* Top Brand / Location Bar */}
          <div className="relative z-10 w-full pt-8 px-8 flex justify-between items-center text-xs tracking-[0.25em] text-zinc-400 uppercase font-mono">
            <span className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-pink-500 animate-ping" />
              <span>CLIZZER AI SYSTEM</span>
            </span>
            <span className="text-amber-400 font-bold">LOS SANTOS ARCHIVE</span>
          </div>

          {/* Center Content: Spinning Radar & Typewriter Text */}
          <div className="relative z-10 flex flex-col items-center justify-center gap-8 px-6 text-center max-w-xl">
            {/* Spinning/Pulsing GTA Radar Shape */}
            <div className="relative w-36 h-36 flex items-center justify-center">
              {/* Outer Counter-Rotating Segmented Ring */}
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-0 rounded-full border-2 border-dashed border-pink-500/50 shadow-[0_0_15px_rgba(236,72,153,0.3)]"
              />

              {/* Middle Clockwise Rotating Ring with Gold Accents */}
              <motion.div
                animate={{ rotate: -360 }}
                transition={{ duration: 5, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-3 rounded-full border-2 border-t-[#f5b800] border-r-transparent border-b-[#f5b800]/40 border-l-transparent shadow-[0_0_12px_rgba(245,184,0,0.4)]"
              />

              {/* Inner Pulsing Radar Core */}
              <motion.div
                animate={{
                  scale: [0.85, 1.15, 0.85],
                  opacity: [0.7, 1, 0.7],
                }}
                transition={{
                  duration: 1.8,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
                className="relative w-16 h-16 rounded-full bg-gradient-to-tr from-pink-600/80 to-[#f5b800]/80 flex items-center justify-center shadow-[0_0_25px_rgba(236,72,153,0.6)]"
              >
                {/* Central Star Icon */}
                <svg
                  className="w-8 h-8 text-black fill-current drop-shadow-md"
                  viewBox="0 0 24 24"
                >
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                </svg>
              </motion.div>

              {/* Crosshair Lines */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-40">
                <div className="w-full h-[1px] bg-pink-400" />
                <div className="absolute h-full w-[1px] bg-pink-400" />
              </div>
            </div>

            {/* Typewriter "LOADING..." Header */}
            <div className="flex flex-col items-center gap-3">
              <div className="flex items-center justify-center">
                <div className="flex">
                  {loadingCharacters.map((char, index) => (
                    <motion.span
                      key={`${char}-${index}`}
                      custom={index}
                      variants={letterVariants}
                      initial="hidden"
                      animate="visible"
                      className="text-3xl sm:text-4xl md:text-5xl font-black italic tracking-widest uppercase text-transparent bg-clip-text bg-gradient-to-r from-pink-500 via-rose-400 to-[#f5b800] drop-shadow-[0_4px_12px_rgba(236,72,153,0.5)]"
                    >
                      {char}
                    </motion.span>
                  ))}
                </div>

                {/* Blinking Block Cursor */}
                <motion.span
                  animate={{ opacity: [0, 1, 0] }}
                  transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
                  className="inline-block ml-1 w-3 sm:w-4 h-7 sm:h-9 bg-[#f5b800] shadow-[0_0_8px_#f5b800]"
                />
              </div>

              {/* Sub-message / Dispatch text */}
              <motion.p
                key={message}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="text-zinc-300 text-sm sm:text-base font-mono tracking-wider uppercase max-w-md px-4 py-1.5 rounded bg-zinc-950/70 border border-zinc-800"
              >
                {message || 'INITIALIZING CLIZZER ENGINE...'}
              </motion.p>
            </div>
          </div>

          {/* Bottom Dock: GTA Progress Bar */}
          <div className="relative z-10 w-full max-w-2xl px-6 pb-10 flex flex-col gap-2">
            <div className="flex justify-between items-end text-xs font-mono font-bold tracking-widest text-zinc-400 uppercase">
              <span className="flex items-center gap-1.5">
                <span className="text-pink-500">▶</span>
                <span>DATA TRANSMISSION</span>
              </span>
              <span className="text-[#f5b800] text-sm">
                {clampedProgress}%
              </span>
            </div>

            {/* Gradient Progress Bar Container */}
            <div className="relative w-full h-3 sm:h-3.5 bg-zinc-900/90 rounded-full border border-zinc-700/60 overflow-hidden shadow-inner p-0.5">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${clampedProgress}%` }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                className="h-full rounded-full bg-gradient-to-r from-pink-500 via-rose-500 to-[#f5b800] shadow-[0_0_15px_rgba(245,184,0,0.65)]"
              />
            </div>
          </div>

          {/* Persistent Hairline Bar at Edge */}
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-zinc-900">
            <motion.div
              animate={{ width: `${clampedProgress}%` }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
              className="h-full bg-gradient-to-r from-pink-500 to-[#f5b800]"
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default GTALoader;
