import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        gtaPink: '#ff007f',
        gtaCyan: '#00f0ff',
        gtaGold: '#f5b800',
        gtaDark: '#0b0b0b',
        gtaPanel: '#1a1a2e',
      },
      fontFamily: {
        gta: ['var(--font-big-shoulders)', 'sans-serif'],
        ops: ['var(--font-black-ops)', 'sans-serif'],
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        glitchText: {
          '0%, 100%': { transform: 'translate(0)' },
          '20%': { transform: 'translate(-2px, 2px)' },
          '40%': { transform: 'translate(-2px, -2px)' },
          '60%': { transform: 'translate(2px, 2px)' },
          '80%': { transform: 'translate(2px, -2px)' },
        },
        wantedPulse: {
          '0%, 100%': {
            transform: 'scale(1)',
            filter: 'drop-shadow(0 0 8px rgba(245, 184, 0, 0.75))',
          },
          '50%': {
            transform: 'scale(1.2)',
            filter: 'drop-shadow(0 0 24px rgba(255, 0, 127, 0.95)) drop-shadow(0 0 12px rgba(245, 184, 0, 0.9))',
          },
        },
      },
      animation: {
        fadeInUp: 'fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        glitchText: 'glitchText 0.3s ease-in-out infinite',
        wantedPulse: 'wantedPulse 1.2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};

export default config;
