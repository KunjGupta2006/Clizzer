'use client';

import React from 'react';
import { motion } from 'framer-motion';

export interface WantedStarsProps {
  /** Wanted level from 0 to 5 */
  level: number;
  /** Whether filled stars should pulse with adrenaline */
  animated?: boolean;
  /** Size preset of the stars */
  size?: 'sm' | 'md' | 'lg';
  /** Optional additional CSS class names */
  className?: string;
}

const sizeConfig = {
  sm: {
    container: 'gap-1',
    svg: 'w-4 h-4',
  },
  md: {
    container: 'gap-1.5',
    svg: 'w-6 h-6',
  },
  lg: {
    container: 'gap-2',
    svg: 'w-8 h-8',
  },
};

export const WantedStars: React.FC<WantedStarsProps> = ({
  level,
  animated = false,
  size = 'md',
  className = '',
}) => {
  // Clamp level to range [0, 5] and round to nearest integer for discrete 5-star display
  const clampedLevel = Math.max(0, Math.min(5, Math.round(level)));
  const { container, svg } = sizeConfig[size] || sizeConfig.md;

  return (
    <div
      className={`inline-flex items-center ${container} ${className}`}
      role="img"
      aria-label={`Wanted level ${clampedLevel} out of 5 stars`}
    >
      {[1, 2, 3, 4, 5].map((starIndex) => {
        const isFilled = starIndex <= clampedLevel;

        const starElement = (
          <svg
            key={starIndex}
            viewBox="0 0 24 24"
            className={`${svg} transition-colors duration-200`}
            fill={isFilled ? '#f5b800' : '#374151'}
            stroke={isFilled ? '#d97706' : '#1f2937'}
            strokeWidth="1"
            strokeLinecap="round"
            strokeLinejoin="round"
            xmlns="http://www.w3.org/2000/svg"
          >
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
        );

        if (isFilled && animated) {
          return (
            <motion.div
              key={starIndex}
              animate={{
                scale: [1, 1.25, 1],
                filter: [
                  'drop-shadow(0 0 2px rgba(245, 184, 0, 0.4))',
                  'drop-shadow(0 0 8px rgba(245, 184, 0, 0.95))',
                  'drop-shadow(0 0 2px rgba(245, 184, 0, 0.4))',
                ],
              }}
              transition={{
                duration: 0.9,
                repeat: Infinity,
                ease: 'easeInOut',
                delay: (starIndex - 1) * 0.12,
              }}
              className="origin-center"
            >
              {starElement}
            </motion.div>
          );
        }

        return (
          <div
            key={starIndex}
            className={isFilled ? 'drop-shadow-[0_0_4px_rgba(245,184,0,0.5)]' : ''}
          >
            {starElement}
          </div>
        );
      })}
    </div>
  );
};

export default WantedStars;
