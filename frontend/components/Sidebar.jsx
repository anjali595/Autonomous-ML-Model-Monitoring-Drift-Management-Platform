import { motion, AnimatePresence } from 'framer-motion';
import { Home, Database, TrendingUp, AlertTriangle, Layers, Upload, FolderOpen, User, Brain } from 'lucide-react';
import { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';

const navItems = [
  { label: 'Dashboard', icon: Home, href: '/' },
  { label: 'Models', icon: Database, href: '/models' },
  { label: 'Upload Model', icon: Upload, href: '/upload-model' },
  { label: 'Predict', icon: Brain, href: '/predict' },
  { label: 'Monitoring', icon: TrendingUp, href: '/monitoring' },
  { label: 'Alerts', icon: AlertTriangle, href: '/alerts' },
  { label: 'Versions', icon: Layers, href: '/versions' },
  { label: 'Datasets', icon: FolderOpen, href: '/datasets' },
  { label: 'Account', icon: User, href: '/account' },
];

export default function Sidebar({ collapsed: initialCollapsed = false }) {
  const [collapsed, setCollapsed] = useState(initialCollapsed);
  const router = useRouter();

  return (
    <motion.aside
      initial={{ width: 260 }}
      animate={{ width: collapsed ? 76 : 260 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="h-screen sticky top-0 left-0 z-30 flex flex-col bg-card border-r border-border"
    >
      {/* Logo */}
      <div className="flex items-center justify-between px-5 py-5 border-b border-border h-[73px]">
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              className="flex items-center gap-3"
            >
              <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                <Brain className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-bold text-foreground tracking-tight">AutoML Drift</span>
            </motion.div>
          )}
        </AnimatePresence>
        <button
          className="rounded-lg p-2 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
          onClick={() => setCollapsed((c) => !c)}
          aria-label="Toggle sidebar"
        >
          <motion.div whileTap={{ scale: 0.8 }}>
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              {collapsed ? (
                <>
                  <path d="M5 12h14" />
                  <path d="m12 5 7 7-7 7" />
                </>
              ) : (
                <>
                  <path d="M19 12H5" />
                  <path d="m12 19-7-7 7-7" />
                </>
              )}
            </svg>
          </motion.div>
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col gap-1 mt-4 px-3 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = router.pathname === item.href;
          return (
            <Link key={item.label} href={item.href} passHref legacyBehavior>
              <motion.a
                whileHover={{ x: 4 }}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-primary text-primary-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                }`}
              >
                <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-primary-foreground' : 'text-muted-foreground'}`} />
                <AnimatePresence>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: 'auto' }}
                      exit={{ opacity: 0, width: 0 }}
                      className="whitespace-nowrap overflow-hidden"
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </motion.a>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-border">
        <AnimatePresence>
          {!collapsed ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-xs font-medium text-muted-foreground"
            >
              © 2026 AutoML Drift
            </motion.div>
          ) : (
            <motion.div className="w-2 h-2 rounded-full bg-border mx-auto" />
          )}
        </AnimatePresence>
      </div>
    </motion.aside>
  );
}
