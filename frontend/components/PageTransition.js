import { motion } from 'framer-motion';

const variants = {
  hidden:  { opacity: 0, y: 16 },
  enter:   { opacity: 1, y: 0 },
  exit:    { opacity: 0, y: -16 },
};

export default function PageTransition({ children }) {
  return (
    <motion.div
      variants={variants}
      initial="hidden"
      animate="enter"
      exit="exit"
      transition={{ duration: 0.28, ease: 'easeOut' }}
    >
      {children}
    </motion.div>
  );
}
