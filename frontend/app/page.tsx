'use client';

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Flame, Play, ShieldCheck, Zap, Sparkles, Video, Volume2, Smile, ArrowRight, Github } from 'lucide-react';
import { WantedStars } from '../components/WantedStars';

export default function LandingPage() {
  return (
    <div className="relative min-h-screen flex flex-col justify-between overflow-hidden">
      {/* Dynamic Background Gradients */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-[#ff007f]/20 rounded-full blur-[128px] pointer-events-none" />
      <div className="absolute bottom-10 right-1/4 w-96 h-96 bg-[#00f0ff]/15 rounded-full blur-[128px] pointer-events-none" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[#f5b800]/10 rounded-full blur-[160px] pointer-events-none" />

      {/* Navigation Header */}
      <header className="relative z-20 flex items-center justify-between px-6 sm:px-12 py-6 border-b border-[#ff007f]/20 bg-black/40 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-tr from-[#ff007f] to-[#f5b800] rounded-lg shadow-[0_0_15px_rgba(255,0,127,0.5)]">
            <Flame className="w-6 h-6 text-black" />
          </div>
          <span className="text-3xl font-black tracking-widest text-[#f5b800] uppercase font-ops drop-shadow-[0_0_10px_rgba(245,184,0,0.6)]">
            CLIZZER
          </span>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1 bg-black/60 border border-[#f5b800]/30 rounded-full">
            <span className="text-xs text-gray-400 font-bold uppercase tracking-wider">HYPE METER:</span>
            <WantedStars level={5} size="sm" animated={true} />
          </div>

          <Link
            href="/clip"
            className="px-5 py-2.5 bg-gradient-to-r from-[#ff007f] to-[#f5b800] hover:from-[#f5b800] hover:to-[#00f0ff] text-black font-black uppercase text-sm tracking-wider rounded-lg transition-all duration-200 transform hover:scale-105 shadow-[0_0_15px_rgba(255,0,127,0.4)]"
          >
            LAUNCH APP
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center text-center px-4 sm:px-8 py-16 max-w-5xl mx-auto">
        {/* GTA Badge */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#1a1a2e] border border-[#ff007f]/60 text-xs sm:text-sm font-extrabold tracking-widest text-[#00f0ff] uppercase shadow-[0_0_15px_rgba(0,240,255,0.25)] mb-6"
        >
          <Sparkles className="w-4 h-4 text-[#f5b800]" />
          <span>VICE CITY AUTOMATED CLIPPING ENGINE</span>
        </motion.div>

        {/* Main Title */}
        <motion.h1
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-6xl sm:text-8xl lg:text-9xl font-black uppercase tracking-tight leading-none"
        >
          <span className="block text-transparent bg-clip-text bg-gradient-to-r from-[#ff007f] via-[#f5b800] to-[#00f0ff] drop-shadow-[0_0_35px_rgba(255,0,127,0.4)] font-ops">
            CLIZZER
          </span>
        </motion.h1>

        {/* Tagline */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-6 text-xl sm:text-3xl text-gray-200 font-extrabold tracking-wider max-w-3xl uppercase font-gta"
        >
          PASTE A LINK. SCAN THE FREQUENCIES.{' '}
          <span className="text-[#f5b800]">GET FIRE VIRAL CLIPS.</span>
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-3 text-sm sm:text-base text-gray-400 max-w-xl"
        >
          100% Free & Open-Source. No GPU Required. Multi-modal hype detection powered by audio frequency analysis, facial emotion tracking, and scene change intelligence.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-5"
        >
          <Link
            href="/clip"
            className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-[#ff007f] via-[#f5b800] to-[#00f0ff] hover:from-[#f5b800] hover:to-[#00f0ff] text-black font-black text-xl tracking-widest uppercase rounded-xl transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-[0_0_30px_rgba(255,0,127,0.6)]"
          >
            <span>START MISSION</span>
            <ArrowRight className="w-6 h-6" />
          </Link>

          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2.5 px-6 py-4 bg-[#1a1a2e] hover:bg-[#252542] border border-gray-700 hover:border-[#f5b800] text-gray-200 font-bold text-base tracking-wider uppercase rounded-xl transition-all duration-200"
          >
            <Github className="w-5 h-5 text-gray-300" />
            <span>OPEN SOURCE REPO</span>
          </a>
        </motion.div>

        {/* 3 Pillars / Feature Highlights */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 w-full text-left"
        >
          <div className="p-6 bg-[#1a1a2e]/70 border border-[#ff007f]/40 rounded-xl backdrop-blur-sm hover:border-[#ff007f] transition-all">
            <Volume2 className="w-8 h-8 text-[#ff007f] mb-3" />
            <h3 className="text-lg font-black text-[#f5b800] uppercase tracking-wider">
              Audio Frequency Hype
            </h3>
            <p className="text-xs text-gray-400 mt-2 leading-relaxed">
              Detects volume spikes, bass drops, crowd cheers, and percussive peaks via Librosa & SciPy.
            </p>
          </div>

          <div className="p-6 bg-[#1a1a2e]/70 border border-[#00f0ff]/40 rounded-xl backdrop-blur-sm hover:border-[#00f0ff] transition-all">
            <Smile className="w-8 h-8 text-[#00f0ff] mb-3" />
            <h3 className="text-lg font-black text-[#f5b800] uppercase tracking-wider">
              Face Mesh & Emotion
            </h3>
            <p className="text-xs text-gray-400 mt-2 leading-relaxed">
              MediaPipe 468-point mesh tracks jaw drops, smiles, and centers faces for 9:16 vertical reframing.
            </p>
          </div>

          <div className="p-6 bg-[#1a1a2e]/70 border border-[#f5b800]/40 rounded-xl backdrop-blur-sm hover:border-[#f5b800] transition-all">
            <Zap className="w-8 h-8 text-[#f5b800] mb-3" />
            <h3 className="text-lg font-black text-[#f5b800] uppercase tracking-wider">
              Zero Subscription Trap
            </h3>
            <p className="text-xs text-gray-400 mt-2 leading-relaxed">
              No minute credit burn, no watermarks, no cloud rate limits. Run locally or via free Colab/Kaggle GPUs.
            </p>
          </div>
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-6 border-t border-gray-900 bg-black/60 text-center text-xs text-gray-500 uppercase tracking-widest font-semibold">
        <p>CLIZZER © 2024 — GTA VICE CITY INSPIRED OPEN-SOURCE CLIPPER • MIT LICENSE</p>
      </footer>
    </div>
  );
}
