import '../styles/globals.css';
import { AnimatePresence, motion } from 'framer-motion';
import { ThemeProvider } from 'next-themes';
import { Toaster } from 'react-hot-toast';

// Pages that should NOT have the sidebar/navbar layout
const authPages = ['/login', '/signup'];

function MyApp({ Component, pageProps, router }) {
  const isAuthPage = authPages.includes(router.pathname);

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1e293b',
            color: '#f8fafc',
            border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '12px',
          },
        }}
      />
      <AnimatePresence mode="wait">
        <motion.div
          key={router.route}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
        >
          <Component {...pageProps} />
        </motion.div>
      </AnimatePresence>
    </ThemeProvider>
  );
}

export default MyApp;
