import { motion } from 'framer-motion';

export default function DriftIndicator({ value = 0, status = 'Normal' }) {
  let color, glow;
  if (status === 'Normal') {
    color = 'stroke-green-400';
    glow = 'shadow-green-400/40';
  } else if (status === 'Warning') {
    color = 'stroke-yellow-400';
    glow = 'shadow-yellow-400/40';
  } else {
    color = 'stroke-red-500';
    glow = 'shadow-red-500/40';
  }
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.7, type: 'spring' }}
      className={`relative flex flex-col items-center justify-center p-6 bg-gradient-to-br from-blue-900/60 to-gray-900/60 rounded-2xl shadow-xl backdrop-blur-xl glass-card ${glow}`}
    >
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="50" stroke="#334155" strokeWidth="12" fill="none" />
        <motion.circle
          cx="60" cy="60" r="50"
          strokeWidth="12"
          fill="none"
          className={color}
          strokeDasharray={314}
          strokeDashoffset={314 - 314 * value}
          strokeLinecap="round"
          initial={{ strokeDashoffset: 314 }}
          animate={{ strokeDashoffset: 314 - 314 * value }}
          transition={{ duration: 1.2, ease: 'easeInOut' }}
          filter={status !== 'Normal' ? 'url(#glow)' : ''}
        />
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-3xl font-bold text-white drop-shadow-lg">
        {(value * 100).toFixed(1)}%
      </div>
      <div className={`mt-4 text-lg font-semibold ${color}`}>{status}</div>
    </motion.div>
  );
}
